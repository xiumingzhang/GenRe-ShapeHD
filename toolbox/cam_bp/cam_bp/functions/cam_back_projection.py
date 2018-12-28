import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from .._ext import cam_bp_lib
from cffi import FFI
ffi = FFI()


class CameraBackProjection(Function):

    @staticmethod
    def forward(ctx, depth_t, fl, cam_dist, res=128):
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
        tdf = depth_t.new(in_shape[0], in_shape[1],
                          res, res, res).zero_() + 1 / res
        cam_bp_lib.back_projection_forward(depth_t, cam_dist, fl, tdf, cnt)
        # print(cnt)
        ctx.save_for_backward(depth_t, fl, cam_dist)
        ctx.cnt_forward = cnt
        ctx.depth_shape = in_shape
        return tdf

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        # print(grad_output.type())
        depth_t, fl, cam_dist = ctx.saved_tensors
        cnt = ctx.cnt_forward
        grad_depth = grad_output.new(ctx.depth_shape).zero_()
        grad_fl = grad_output.new(
            ctx.depth_shape[0], ctx.depth_shape[1]).zero_()
        grad_camdist = grad_output.new(
            ctx.depth_shape[0], ctx.depth_shape[1]).zero_()
        cam_bp_lib.back_projection_backward(
            depth_t, fl, cam_dist, cnt, grad_output, grad_depth, grad_camdist, grad_fl)
        return grad_depth, grad_fl, grad_camdist, None
