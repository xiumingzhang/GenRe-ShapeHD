import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from .._ext import calc_prob_lib
from cffi import FFI
ffi = FFI()


class CalcStopProb(Function):
    @staticmethod
    def forward(ctx, prob_in):
        assert prob_in.dim() == 5
        assert prob_in.dtype == torch.float32
        assert prob_in.is_cuda
        stop_prob = prob_in.new(prob_in.shape)
        stop_prob.zero_()
        calc_prob_lib.calc_prob_forward(prob_in, stop_prob)
        ctx.save_for_backward(prob_in, stop_prob)
        return stop_prob

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_in):
        prob_in, stop_prob = ctx.saved_tensors
        grad_out = grad_in.new(grad_in.shape)
        grad_out.zero_()
        stop_prob_weighted = stop_prob * grad_in
        calc_prob_lib.calc_prob_backward(prob_in, stop_prob_weighted, grad_out)
        return grad_out
