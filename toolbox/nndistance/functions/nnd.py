# functions/add.py
import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from nndistance._ext import my_lib


class NNDFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        assert xyz1.dim() == 3 and xyz2.dim() == 3
        assert xyz1.size(0) == xyz2.size(0)
        assert xyz1.size(2) == 3 and xyz2.size(2) == 3
        assert xyz1.is_cuda == xyz2.is_cuda
        assert xyz1.type().endswith('FloatTensor') and xyz2.type().endswith('FloatTensor'), 'only FloatTensor are supported for NNDistance'
        assert xyz1.is_contiguous() and xyz2.is_contiguous()  # the CPU and GPU code are not robust and will break if the storage is not contiguous
        ctx.is_cuda = xyz1.is_cuda

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()
        dist1 = torch.zeros(batchsize, n)
        dist2 = torch.zeros(batchsize, m)

        idx1 = torch.zeros(batchsize, n).type(torch.IntTensor)
        idx2 = torch.zeros(batchsize, m).type(torch.IntTensor)

        if not xyz1.is_cuda:
            my_lib.nnd_forward(xyz1, xyz2, dist1, dist2, idx1, idx2)
        else:
            dist1 = dist1.cuda()
            dist2 = dist2.cuda()
            idx1 = idx1.cuda()
            idx2 = idx2.cuda()
            my_lib.nnd_forward_cuda(xyz1, xyz2, dist1, dist2, idx1, idx2)

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    @once_differentiable
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        """
        Note that this function needs gradidx placeholders
        """
        assert ctx.is_cuda == graddist1.is_cuda and ctx.is_cuda == graddist2.is_cuda
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        graddist1 = graddist1.contiguous()
        graddist2 = graddist2.contiguous()
        assert xyz1.is_contiguous()
        assert xyz2.is_contiguous()
        assert idx1.is_contiguous()
        assert idx2.is_contiguous()
        assert graddist1.type().endswith('FloatTensor') and graddist2.type().endswith('FloatTensor'), 'only FloatTensor are supported for NNDistance'

        gradxyz1 = xyz1.new(xyz1.size())
        gradxyz2 = xyz1.new(xyz2.size())

        if not graddist1.is_cuda:
            my_lib.nnd_backward(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)
        else:
            my_lib.nnd_backward_cuda(xyz1, xyz2, gradxyz1, gradxyz2, graddist1, graddist2, idx1, idx2)

        return gradxyz1, gradxyz2


def nndistance_w_idx(xyz1, xyz2):
    xyz1 = xyz1.contiguous()
    xyz2 = xyz2.contiguous()
    return NNDFunction.apply(xyz1, xyz2)


def nndistance(xyz1, xyz2):
    if xyz1.size(2) != 3:
        xyz1 = xyz1.transpose(1, 2)
    if xyz2.size(2) != 3:
        xyz2 = xyz2.transpose(1, 2)
    xyz1 = xyz1.contiguous()
    xyz2 = xyz2.contiguous()
    dist1, dist2, _, _ = NNDFunction.apply(xyz1, xyz2)
    return dist1, dist2


def nndistance_score(xyz1, xyz2, eps=1e-10):
    dist1, dist2 = nndistance(xyz1, xyz2)
    return torch.sqrt(dist1 + eps).mean(1) + torch.sqrt(dist2 + eps).mean(1)
