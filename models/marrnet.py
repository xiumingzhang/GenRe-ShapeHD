from os import makedirs
from os.path import join
import numpy as np
import torch
import torch.nn as nn
from util import util_img
from .marrnet1 import Net as Marrnet1
from .marrnet2 import Net as Marrnet2, Model as Marrnet2_model


class Model(Marrnet2_model):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument(
            '--canon_sup',
            action='store_true',
            help="Use canonical-pose voxels as supervision"
        )
        parser.add_argument(
            '--marrnet1',
            type=str, default=None,
            help="Path to pretrained MarrNet-1"
        )
        parser.add_argument(
            '--marrnet2',
            type=str, default=None,
            help="Path to pretrained MarrNet-2 (to be finetuned)"
        )
        return parser, set()

    def __init__(self, opt, logger):
        super().__init__(opt, logger)
        pred_silhou_thres = self.pred_silhou_thres * self.scale_25d
        self.requires = ['rgb', self.voxel_key]
        self.net = Net(opt.marrnet1, opt.marrnet2, pred_silhou_thres)
        self._nets = [self.net]
        self.optimizer = self.adam(
            self.net.marrnet2.parameters(),
            lr=opt.lr,
            **self.optim_params
        ) # just finetune MarrNet-2
        self._optimizers[-1] = self.optimizer
        self.input_names = ['rgb']
        self.init_vars(add_path=True)

    def __str__(self):
        return "Finetuning MarrNet-2 with MarrNet-1 predictions"

    def pack_output(self, pred, batch, add_gt=True):
        pred_normal = pred['normal'].detach().cpu()
        pred_silhou = pred['silhou'].detach().cpu()
        pred_depth = pred['depth'].detach().cpu()
        out = {}
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
        out['pred_voxel'] = pred['voxel'].detach().cpu().numpy()
        if add_gt:
            out['gt_voxel'] = batch[self.voxel_key].numpy()
        return out

    def compute_loss(self, pred):
        loss = self.criterion(
            pred['voxel'],
            getattr(self._gt, self.voxel_key)
        )
        loss_data = {}
        loss_data['loss'] = loss.mean().item()
        return loss, loss_data


class Net(nn.Module):
    """
       MarrNet-1    MarrNet-2
    RGB ------> 2.5D ------> 3D
         fixed      finetuned
    """

    def __init__(self, marrnet1_path=None, marrnet2_path=None, pred_silhou_thres=0.3):
        super().__init__()
        # Init MarrNet-1 and load weights
        self.marrnet1 = Marrnet1(
            [3, 1, 1],
            ['normal', 'depth', 'silhou'],
            pred_depth_minmax=True, # not used in MarrNet
        )
        if marrnet1_path:
            state_dict = torch.load(marrnet1_path)['nets'][0]
            self.marrnet1.load_state_dict(state_dict)
        # Init MarrNet-2 and load weights
        self.marrnet2 = Marrnet2(4)
        if marrnet2_path:
            state_dict = torch.load(marrnet2_path)['nets'][0]
            self.marrnet2.load_state_dict(state_dict)
        # Fix MarrNet-1, but finetune 2
        for p in self.marrnet1.parameters():
            p.requires_grad = False
        for p in self.marrnet2.parameters():
            p.requires_grad = True
        self.pred_silhou_thres = pred_silhou_thres

    def forward(self, input_struct):
        # Predict 2.5D sketches
        with torch.no_grad():
            pred = self.marrnet1(input_struct)
        depth = pred['depth']
        normal = pred['normal']
        silhou = pred['silhou']
        # Mask
        is_bg = silhou < self.pred_silhou_thres
        depth[is_bg] = 0
        normal[is_bg.repeat(1, 3, 1, 1)] = 0
        x = torch.cat((depth, normal), 1)
        # Forward
        latent_vec = self.marrnet2.encoder(x)
        vox = self.marrnet2.decoder(latent_vec)
        pred['voxel'] = vox
        return pred


class Model_test(Model):
    def __init__(self, opt, logger):
        super().__init__(opt, logger)
        self.requires = ['rgb', 'mask'] # mask for bbox cropping only
        self.load_state_dict(opt.net_file, load_optimizer='auto')
        self.input_names = ['rgb']
        self.init_vars(add_path=True)
        self.output_dir = opt.output_dir

    def __str__(self):
        return "Testing MarrNet"

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
        pred = self.predict(batch, load_gt=False, no_grad=True)
        output = self.pack_output(pred, batch, add_gt=False)
        self.visualizer.visualize(output, batch_i, outdir)
        np.savez(outdir + '.npz', **output)
