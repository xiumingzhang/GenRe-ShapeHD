from torch import nn
from ..functions import CameraBackProjection
import torch


class Camera_back_projection_layer(nn.Module):
    def __init__(self, res=128):
        super(Camera_back_projection_layer, self).__init__()
        assert res == 128
        self.res = 128

    def forward(self, depth_t, fl=418.3, cam_dist=2.2, shift=True):
        n = depth_t.size(0)
        if type(fl) == float:
            fl_v = fl
            fl = torch.FloatTensor(n, 1).cuda()
            fl.fill_(fl_v)
        if type(cam_dist) == float:
            cmd_v = cam_dist
            cam_dist = torch.FloatTensor(n, 1).cuda()
            cam_dist.fill_(cmd_v)
        df = CameraBackProjection.apply(depth_t, fl, cam_dist, self.res)
        return self.shift_tdf(df) if shift else df

    @staticmethod
    def shift_tdf(input_tdf, res=128):
        out_tdf = 1 - res * (input_tdf)
        return out_tdf


class camera_backprojection(nn.Module):

    def __init__(self, vox_res=128):
        super(camera_backprojection, self).__init__()
        self.vox_res = vox_res
        self.backprojection_layer = CameraBackProjection()

    def forward(self, depth, fl, camdist):
        return self.backprojection_layer(depth, fl, camdist, self.voxel_res)
