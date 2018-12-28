import torch
import numpy as np
from .._ext import cam_bp_lib
from cffi import FFI
ffi = FFI()


def get_vox_surface_cnt(depth_t, fl, cam_dist, res=128):
    assert depth_t.dim() == 4
    assert fl.dim() == 2 and fl.size(1) == depth_t.size(1)
    assert cam_dist.dim() == 2 and cam_dist.size(1) == depth_t.size(1)
    assert cam_dist.size(0) == depth_t.size(0)
    assert fl.size(0) == depth_t.size(0)
    assert depth_t.is_cuda
    assert fl.is_cuda
    assert cam_dist.is_cuda
    in_shape = depth_t.shape
    cnt = depth_t.new(in_shape[0], in_shape[1], res, res, res).zero_()
    tdf = depth_t.new(in_shape[0], in_shape[1], res,
                      res, res).zero_() + 1 / res
    cam_bp_lib.back_projection_forward(depth_t, cam_dist, fl, tdf, cnt)
    return cnt


def get_surface_mask(depth_t, fl=784.4645406, cam_dist=2.0, res=128):
    n = depth_t.size(0)
    nc = depth_t.size(1)
    if type(fl) == float:
        fl_v = fl
        fl = torch.FloatTensor(n, nc).cuda()
        fl.fill_(fl_v)
    if type(cam_dist) == float:
        cmd_v = cam_dist
        cam_dist = torch.FloatTensor(n, nc).cuda()
        cam_dist.fill_(cmd_v)
    cnt = get_vox_surface_cnt(depth_t, fl, cam_dist, res)
    mask = cnt.new(n, nc, res, res, res).zero_()
    cam_bp_lib.get_surface_mask(depth_t, cam_dist, fl, cnt, mask)
    surface_vox = torch.clamp(cnt, min=0.0, max=1.0)
    return surface_vox, mask
