import torch
from torch import nn
from ..functions import SphericalBackProjection
from torch.autograd import Variable


class spherical_backprojection(nn.Module):

    def __init__(self, grid, vox_res=128):
        super(camera_backprojection, self).__init__()
        self.vox_res = vox_res
        self.backprojection_layer = SphericalBackProjection()
        assert type(grid) == torch.FloatTensor
        self.grid = Variable(grid.cuda())

    def forward(self, spherical):
        return self.backprojection_layer(spherical, self.grid, self.vox_res)
