import torch
import numpy as np
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from .._ext import cam_bp_lib
from cffi import FFI
ffi = FFI()


class SphericalBackProjection(Function):

    @staticmethod
    def forward(ctx, spherical, grid, res=128):
        assert spherical.dim() == 4
        assert grid.dim() == 5
        assert spherical.size(0) == grid.size(0)
        assert spherical.size(1) == grid.size(1)
        assert spherical.size(2) == grid.size(2)
        assert spherical.size(3) == grid.size(3)
        assert grid.size(4) == 3
        assert spherical.is_cuda
        assert grid.is_cuda
        in_shape = spherical.shape
        cnt = spherical.new(in_shape[0], in_shape[1], res, res, res).zero_()
        tdf = spherical.new(in_shape[0], in_shape[1],
                            res, res, res).zero_()
        cam_bp_lib.spherical_back_proj_forward(spherical, grid, tdf, cnt)
        # print(cnt)
        ctx.save_for_backward(spherical.detach(), grid, cnt)
        ctx.depth_shape = in_shape
        return tdf, cnt

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output, grad_phony):
        assert grad_output.is_cuda
        assert not np.isnan(torch.sum(grad_output.detach()))
        spherical, grid, cnt = ctx.saved_tensors
        grad_depth = grad_output.new(ctx.depth_shape).zero_()
        cam_bp_lib.spherical_back_proj_backward(
            spherical, grid, cnt, grad_output, grad_depth)
        try:
            assert not np.isnan(torch.sum(grad_depth))
        except:
            import pdb
            pdb.set_trace()
        return grad_depth, None, None
