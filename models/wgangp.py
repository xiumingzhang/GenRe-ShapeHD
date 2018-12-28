from os import makedirs
from os.path import join
from time import time
import numpy as np
import torch
from networks.networks import VoxelGenerator, VoxelDiscriminator
from .netinterface import NetInterface


class Model(NetInterface):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            '--canon_voxel',
            action='store_true',
            help="Generate/discriminate canonical-pose voxels"
        )
        parser.add_argument(
            '--wgangp_lambda',
            type=float,
            default=10,
            help="WGANGP gradient penalty coefficient"
        )
        parser.add_argument(
            '--wgangp_norm',
            type=float,
            default=1,
            help="WGANGP gradient penalty norm"
        )
        parser.add_argument(
            '--gan_d_iter',
            type=int,
            default=1,
            help="# iterations D is trained per G's iteration"
        )
        return parser, set()

    def __init__(self, opt, logger):
        super().__init__(opt, logger)
        assert opt.canon_voxel, "GAN requires canonical-pose voxels to work"
        self.requires = ['voxel_canon']
        self.nz = 200
        self.net_g = G(self.nz)
        self.net_d = D()
        self._nets = [self.net_g, self.net_d]
        # Optimizers
        self.optim_params = dict()
        self.optim_params['betas'] = (opt.adam_beta1, opt.adam_beta2)
        self.optimizer_g = self.adam(
            self.net_g.parameters(),
            lr=opt.lr,
            **self.optim_params
        )
        self.optimizer_d = self.adam(
            self.net_d.parameters(),
            lr=opt.lr,
            **self.optim_params
        )
        self._optimizers = [self.optimizer_g, self.optimizer_d]
        #
        self.opt = opt
        self.preprocess = None
        self._metrics = ['err_d_real', 'err_d_fake', 'err_d_gp', 'err_d', 'err_g', 'loss']
        if opt.log_time:
            self._metrics += ['t_d_real', 't_d_fake', 't_d_grad', 't_g']
        self.input_names = ['voxel_canon']
        self.aux_names = ['one', 'neg_one']
        self.init_vars(add_path=True)
        self.init_weight(self.net_d)
        self.init_weight(self.net_g)
        self._last_err_g = None

    def __str__(self):
        s = "3D-WGANGP"
        return s

    def _train_on_batch(self, epoch, batch_idx, batch):
        net_d, net_g = self.net_d, self.net_g
        opt_d, opt_g = self.optimizer_d, self.optimizer_g
        one = self._aux.one
        neg_one = self._aux.neg_one
        real = batch['voxel_canon'].cuda()
        batch_size = real.shape[0]
        batch_log = {'size': batch_size}

        # Train D ...
        net_d.zero_grad()
        for p in net_d.parameters():
            p.requires_grad = True
        for p in net_g.parameters():
            p.requires_grad = False
        # with real
        t0 = time()
        err_d_real = self.net_d(real).mean()
        err_d_real.backward(neg_one)
        batch_log['err_d_real'] = -err_d_real.item()
        d_real_t = time() - t0
        # with fake
        t0 = time()
        with torch.no_grad():
            _, fake = self.net_g(batch_size)
        err_d_fake = self.net_d(fake).mean()
        err_d_fake.backward(one)
        batch_log['err_d_fake'] = err_d_fake.item()
        d_fake_t = time() - t0
        # with grad penalty
        t0 = time()
        if self.opt.wgangp_lambda > 0:
            grad_penalty = self.calc_grad_penalty(real, fake)
            grad_penalty.backward()
            batch_log['err_d_gp'] = grad_penalty.item()
        else:
            batch_log['err_d_gp'] = 0
        batch_log['err_d'] = batch_log['err_d_fake'] + batch_log['err_d_real'] \
            + batch_log['err_d_gp']
        d_grad_t = time() - t0
        opt_d.step()

        # Train G
        t0 = time()
        for p in net_d.parameters():
            p.requires_grad = False
        for p in net_g.parameters():
            p.requires_grad = True
        net_g.zero_grad()
        if batch_idx % self.opt.gan_d_iter == 0:
            _, gen = self.net_g(batch_size)
            err_g = self.net_d(gen).mean()
            err_g.backward(neg_one)
            opt_g.step()
            batch_log['err_g'] = -err_g.item()
            self._last_err_g = batch_log['err_g']
        else:
            batch_log['err_g'] = self._last_err_g
        g_t = time() - t0

        if self.opt.log_time:
            batch_log['t_d_real'] = d_real_t
            batch_log['t_d_fake'] = d_fake_t
            batch_log['t_d_grad'] = d_grad_t
            batch_log['t_g'] = g_t
        return batch_log

    def calc_grad_penalty(self, real, fake):
        alpha = torch.rand(real.shape[0], 1)
        alpha = alpha.expand(
            real.shape[0], real.nelement() // real.shape[0]
        ).contiguous().view(*real.shape).cuda()
        inter = alpha * real + (1 - alpha) * fake
        inter.requires_grad = True
        err_d_inter = self.net_d(inter)
        grads = torch.autograd.grad(
            outputs=err_d_inter,
            inputs=inter,
            grad_outputs=torch.ones(err_d_inter.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grads = grads.view(grads.size(0), -1)
        grad_penalty = (
            ((grads + 1e-16).norm(2, dim=1) - self.opt.wgangp_norm) ** 2
        ).mean() * self.opt.wgangp_lambda
        return grad_penalty

    def _vali_on_batch(self, epoch, batch_idx, batch):
        batch_size = batch['voxel_canon'].shape[0]
        batch_log = {'size': batch_size}
        with torch.no_grad():
            noise, gen = self.net_g(batch_size)
            disc = self.net_d(gen)
        batch_log['loss'] = -disc.mean().item()
        # Save and visualize
        if np.mod(epoch, self.opt.vis_every_train) == 0:
            if batch_idx < self.opt.vis_batches_train:
                outdir = join(self.full_logdir, 'epoch%04d_vali' % epoch)
                makedirs(outdir, exist_ok=True)
                output = self.pack_output(noise, gen, disc)
                self.visualizer.visualize(output, batch_idx, outdir)
                np.savez(join(outdir, 'batch%04d' % batch_idx), **output)
        return batch_log

    @staticmethod
    def pack_output(noise, gen, disc):
        out = {
            'noise': noise.cpu().numpy(),
            'gen_voxel': gen.cpu().numpy(),
            'disc': disc.cpu().numpy(),
        }
        return out


class G(VoxelGenerator):
    def __init__(self, nz):
        super().__init__(nz=nz, nf=64, bias=False, res=128)
        self.nz = nz
        self.noise = torch.FloatTensor().cuda()

    def forward(self, batch_size):
        x = self.noise
        x.resize_(batch_size, self.nz, 1, 1, 1).normal_(0, 1)
        y = super().forward(x)
        return x, y


class D(VoxelDiscriminator):
    def __init__(self):
        super().__init__(nf=64, bias=False, res=128)

    def forward(self, x):
        if x.dim() == 4:
            x.unsqueeze_(1)
        y = super().forward(x)
        return y
