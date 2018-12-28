from os import makedirs
from os.path import join
import numpy as np
import torch
from torch import nn
from models.netinterface import NetInterface
from toolbox.cam_bp.cam_bp.functions import CameraBackProjection
import util.util_img


class MarrnetBaseModel(NetInterface):
    im_size = 256
    rgb_jitter_d = 0.4
    rgb_light_noise = 0.1
    silhou_thres = 0.999
    pred_silhou_thres = 0.3
    scale_25d = 100

    def __init__(self, opt, logger):
        super(MarrnetBaseModel, self).__init__(opt, logger)
        self.opt = opt
        self.n_batches_per_epoch = opt.epoch_batches
        self.n_batches_to_vis_train = opt.vis_batches_train
        self.n_batches_to_vis_vali = opt.vis_batches_vali
        self.full_logdir = opt.full_logdir
        self._metrics = []
        self.batches_to_vis = {}
        self.dataset = opt.dataset
        self._nets = []
        self._optimizers = []
        self._moveable_vars = []
        self.cam_bp = Camera_back_projection_layer(128)
        if opt.log_time:
            self._metrics += ['batch_time', 'data_time']
        # Parameters for different optimization methods
        self.optim_params = dict()
        if opt.optim == 'adam':
            self.optim_params['betas'] = (opt.adam_beta1, opt.adam_beta2)
        elif opt.optim == 'sgd':
            self.optim_params['momentum'] = opt.sgd_momentum
            self.optim_params['dampening'] = opt.sgd_dampening
            self.optim_params['weight_decay'] = opt.sgd_wdecay
        else:
            raise NotImplementedError(opt.optim)

    def _train_on_batch(self, batch_idx, batch):
        self.net.zero_grad()
        pred = self.predict(batch)
        loss, loss_data = self.compute_loss(pred)
        loss.backward()
        self.optimizer.step()
        batch_size = len(batch['rgb_path'])
        batch_log = {'size': batch_size, **loss_data}
        self.record_batch(batch_idx, batch)
        return batch_log

    def _vali_on_batch(self, epoch, batch_idx, batch):
        pred = self.predict(batch, no_grad=True)
        _, loss_data = self.compute_loss(pred)
        if np.mod(epoch, self.opt.vis_every_vali) == 0:
            if batch_idx < self.opt.vis_batches_vali:
                outdir = join(self.full_logdir, 'epoch%04d_vali' % epoch)
                makedirs(outdir, exist_ok=True)
                output = self.pack_output(pred, batch)
                self.visualizer.visualize(output, batch_idx, outdir)
                np.savez(join(outdir, 'batch%04d' % batch_idx), **output)
        batch_size = len(batch['rgb_path'])
        batch_log = {'size': batch_size, **loss_data}
        return batch_log

    @classmethod
    def preprocess(cls, data, mode='train'):
        """
        This function should be applied to [0, 1] floats, except absolute depth
        """
        data_proc = {}
        for key, val in data.items():
            if key == 'rgb':
                im = val
                # H x W x 3
                im = util.util_img.resize(im, cls.im_size, 'horizontal')
                if mode == 'train':
                    im = util.util_img.jitter_colors(
                        im,
                        d_brightness=cls.rgb_jitter_d,
                        d_contrast=cls.rgb_jitter_d,
                        d_saturation=cls.rgb_jitter_d
                    )
                    im = util.util_img.add_lighting_noise(
                        im, cls.rgb_light_noise)
                im = util.util_img.normalize_colors(im)
                val = im.transpose(2, 0, 1)

            elif key == 'depth':
                im = val
                if im.ndim == 3:
                    im = im[:, :, 0]
                im = util.util_img.resize(
                    im, cls.im_size, 'horizontal', clamp=(im.min(), im.max()))
                im *= cls.scale_25d
                val = im[np.newaxis, :, :]
                # 1 x H x W, scaled

            elif key == 'silhou':
                im = val
                if im.ndim == 3:
                    im = im[:, :, 0]
                im = util.util_img.resize(
                    im, cls.im_size, 'horizontal', clamp=(im.min(), im.max()))
                im = util.util_img.binarize(
                    im, cls.silhou_thres, gt_is_1=True)
                im *= cls.scale_25d
                val = im[np.newaxis, :, :]
                # 1 x H x W, binarized, scaled

            elif key == 'normal':
                # H x W x 3
                im = val
                im = util.util_img.resize(
                    im, cls.im_size, 'horizontal', clamp=(im.min(), im.max()))
                im *= cls.scale_25d
                val = im.transpose(2, 0, 1)
                # 3 x H x W, scaled

            data_proc[key] = val
        return data_proc

    @staticmethod
    def mask(input_image, input_mask, bg=1.0):
        assert isinstance(bg, (int, float))
        assert (input_mask >= 0).all() and (input_mask <= 1).all()
        input_mask = input_mask.expand_as(input_image)
        bg = bg * input_image.new_ones(input_image.size())
        output = input_mask * input_image + (1 - input_mask) * bg
        return output

    @classmethod
    def postprocess(cls, tensor, bg=1.0, input_mask=None):
        scaled = tensor / cls.scale_25d
        if input_mask is not None:
            return cls.mask(scaled, input_mask, bg=bg)
        return scaled

    @staticmethod
    def to_abs_depth(rel_depth, depth_minmax):
        bmin = depth_minmax[:, 0]
        bmax = depth_minmax[:, 1]
        depth_min = bmin.view(-1, 1, 1, 1)
        depth_max = bmax.view(-1, 1, 1, 1)
        abs_depth = rel_depth * (depth_max - depth_min + 1e-4) + depth_min
        return abs_depth

    def proj_depth(self, abs_depth):
        proj_depth = self.cam_bp(abs_depth)
        return self.cam_bp.shift_tdf(proj_depth)


class Camera_back_projection_layer(nn.Module):
    def __init__(self, res):
        super(Camera_back_projection_layer, self).__init__()
        self.res = res

    def forward(self, depth_t, fl=784.4645406, cam_dist=2.2):
        # print(cam_dist)
        n = depth_t.size(0)
        if isinstance(fl, float):
            fl_v = fl
            fl = torch.FloatTensor(n, 1).cuda()
            fl.fill_(fl_v)
        if isinstance(cam_dist, float):
            cmd_v = cam_dist
            cam_dist = torch.FloatTensor(n, 1).cuda()
            cam_dist.fill_(cmd_v)
        return CameraBackProjection.apply(depth_t, fl, cam_dist, self.res)

    @staticmethod
    def shift_tdf(input_tdf, res=128):
        out_tdf = 1 - res * input_tdf
        return out_tdf
