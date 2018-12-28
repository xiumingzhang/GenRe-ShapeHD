from os import makedirs
from os.path import join
import numpy as np
import torch
import torch.nn as nn
from util import util_img
from .wgangp import D
from .marrnet2 import Net as Marrnet2, Model as Marrnet2_model
from .marrnet1 import Model as Marrnet1_model


class Model(Marrnet2_model):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            '--canon_sup',
            action='store_true',
            help="Use canonical-pose voxels as supervision"
        )
        parser.add_argument(
            '--marrnet2',
            type=str, default=None,
            help="Path to pretrained MarrNet-2 (to be finetuned)"
        )
        parser.add_argument(
            '--gan',
            type=str, default=None,
            help="Path to pretrained WGANGP"
        )
        parser.add_argument(
            '--w_gan_loss',
            type=float, default=0,
            help="Weight for perceptual loss relative to supervised loss"
        )
        return parser, set()

    def __init__(self, opt, logger):
        super().__init__(opt, logger)
        assert opt.canon_sup, "ShapeHD uses canonical-pose voxels"
        self.net = Net(opt.marrnet2, opt.gan)
        self._nets = [self.net]
        self.optimizer = self.adam(
            self.net.marrnet2.parameters(),
            lr=opt.lr,
            **self.optim_params
        ) # just finetune MarrNet-2
        self._optimizers[-1] = self.optimizer
        self._metrics += ['sup', 'gan']
        self.init_vars(add_path=True)
        assert opt.w_gan_loss >= 0

    def __str__(self):
        return "Finetuning 3D estimator of ShapeHD with GAN loss"

    def pack_output(self, pred, batch, add_gt=True):
        out = {}
        out['rgb_path'] = batch['rgb_path']
        out['pred_voxel_noft'] = pred['voxel_noft'].detach().cpu().numpy()
        out['pred_voxel'] = pred['voxel'].detach().cpu().numpy()
        if add_gt:
            out['gt_voxel'] = batch[self.voxel_key].numpy()
            out['normal_path'] = batch['normal_path']
            out['depth_path'] = batch['depth_path']
            out['silhou_path'] = batch['silhou_path']
        return out

    def compute_loss(self, pred):
        loss_sup = self.criterion(
            pred['voxel'], # will be sigmoid'ed
            getattr(self._gt, self.voxel_key)
        )
        loss_gan = -pred['is_real'].mean() # negate to maximize
        loss_gan *= self.opt.w_gan_loss
        loss = loss_sup + loss_gan
        loss_data = {}
        loss_data['sup'] = loss_sup.item()
        loss_data['gan'] = loss_gan.item()
        loss_data['loss'] = loss.item()
        return loss, loss_data


class Net(nn.Module):
    """
       3D Estimator   D of GAN
    2.5D --------> 3D --------> real/fake
         finetuned     fixed
    """

    def __init__(self, marrnet2_path=None, gan_path=None):
        super().__init__()
        # Init MarrNet-2 and load weights
        self.marrnet2 = Marrnet2(4)
        self.marrnet2_noft = Marrnet2(4)
        if marrnet2_path:
            state_dicts = torch.load(marrnet2_path)
            state_dict = state_dicts['nets'][0]
            self.marrnet2.load_state_dict(state_dict)
            self.marrnet2_noft.load_state_dict(state_dict)
        # Init discriminator and load weights
        self.d = D()
        if gan_path:
            state_dicts = torch.load(gan_path)
            self.d.load_state_dict(state_dicts['nets'][1])
        # Fix D, but finetune MarrNet-2
        for p in self.d.parameters():
            p.requires_grad = False
        for p in self.marrnet2_noft.parameters():
            p.requires_grad = False
        for p in self.marrnet2.parameters():
            p.requires_grad = True
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_struct):
        pred = {}
        pred['voxel_noft'] = self.marrnet2_noft(input_struct) # unfinetuned
        pred['voxel'] = self.marrnet2(input_struct)
        pred['is_real'] = self.d(self.sigmoid(pred['voxel']))
        return pred


class Model_test(Model):
    @classmethod
    def add_arguments(cls, parser):
        parser, unique_params = Model.add_arguments(parser)
        parser.add_argument(
            '--marrnet1_file',
            type=str, required=True,
            help="Path to pretrained MarrNet-1"
        )
        return parser, unique_params

    def __init__(self, opt, logger):
        opt.canon_sup = True # dummy, for network init only
        super().__init__(opt, logger)
        self.requires = ['rgb', 'mask'] # mask for bbox cropping only
        self.input_names = ['rgb']
        self.init_vars(add_path=True)
        self.output_dir = opt.output_dir
        # Load MarrNet-2 and D (though unused at test time)
        self.load_state_dict(opt.net_file, load_optimizer='auto')
        # Load MarrNet-1 whose outputs are inputs to D-tuned MarrNet-2
        opt.pred_depth_minmax = True # dummy
        self.marrnet1 = Marrnet1_model(opt, logger)
        self.marrnet1.load_state_dict(opt.marrnet1_file)
        self._nets.append(self.marrnet1.net)

    def __str__(self):
        return "Testing ShapeHD"

    @classmethod
    def preprocess_wrapper(cls, in_dict):
        silhou_thres = 0.95
        in_size = 480
        pad = 85
        im = in_dict['rgb']
        mask = in_dict['silhou']
        bbox = util_img.get_bbox(mask, th=silhou_thres)
        im_crop = util_img.crop(im, bbox, in_size, pad, pad_zero=False)
        in_dict['rgb'] = im_crop
        del in_dict['silhou'] # just for cropping -- done its job
        # Now the image is just like those we rendered
        out_dict = cls.preprocess(in_dict, mode='test')
        return out_dict

    def test_on_batch(self, batch_i, batch):
        outdir = join(self.output_dir, 'batch%04d' % batch_i)
        makedirs(outdir, exist_ok=True)
        # Forward MarrNet-1
        pred1 = self.marrnet1.predict(batch, load_gt=False, no_grad=True)
        # Forward MarrNet-2
        for net_name in ('marrnet2', 'marrnet2_noft'):
            net = getattr(self.net, net_name)
            net.silhou_thres = self.pred_silhou_thres * self.scale_25d
        self.input_names = ['depth', 'normal', 'silhou']
        pred2 = self.predict(pred1, load_gt=False, no_grad=True)
        # Pack, visualize, and save outputs
        output = self.pack_output(pred1, pred2, batch)
        self.visualizer.visualize(output, batch_i, outdir)
        np.savez(outdir + '.npz', **output)

    def pack_output(self, pred1, pred2, batch):
        out = {}
        # MarrNet-1 outputs
        pred_normal = pred1['normal'].detach().cpu()
        pred_silhou = pred1['silhou'].detach().cpu()
        pred_depth = pred1['depth'].detach().cpu()
        out['rgb_path'] = batch['rgb_path']
        out['rgb'] = util_img.denormalize_colors(batch['rgb'].detach().numpy())
        pred_silhou = self.postprocess(pred_silhou)
        pred_silhou = torch.clamp(pred_silhou, 0, 1)
        pred_silhou[pred_silhou < 0] = 0
        out['pred_silhou'] = pred_silhou.numpy()
        out['pred_normal'] = self.postprocess(
            pred_normal, bg=1.0, input_mask=pred_silhou
        ).numpy()
        out['pred_depth'] = self.postprocess(
            pred_depth, bg=0.0, input_mask=pred_silhou
        ).numpy()
        # D-tuned MarrNet-2 outputs
        out['pred_voxel'] = pred2['voxel'].detach().cpu().numpy()
        out['pred_voxel_noft'] = pred2['voxel_noft'].detach().cpu().numpy()
        return out
