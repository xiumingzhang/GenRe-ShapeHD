import numpy as np
import torch
from .calc_prob.calc_prob.functions.calc_prob import CalcStopProb


def gen_sph_grid(res=128):
    pi = np.pi
    phi = np.linspace(0, 180, res * 2 + 1)[1::2]
    theta = np.linspace(0, 360, res + 1)[:-1]
    grid = np.zeros([res, res, 3])
    for idp, p in enumerate(phi):
        for idt, t in enumerate(theta):
            grid[idp, idt, 2] = np.cos((p * pi / 180))
            proj = np.sin((p * pi / 180))
            grid[idp, idt, 0] = proj * np.cos(t * pi / 180)
            grid[idp, idt, 1] = proj * np.sin(t * pi / 180)
    grid = np.reshape(grid, (1, 1, res, res, 3))
    return torch.from_numpy(grid).float()


def sph_pad(sph_tensor, padding_margin=16):
    F = torch.nn.functional
    pad2d = (padding_margin, padding_margin, padding_margin, padding_margin)
    rep_padded_sph = F.pad(sph_tensor, pad2d, mode='replicate')
    _, _, h, w = rep_padded_sph.shape
    rep_padded_sph[:, :, :, 0:padding_margin] = rep_padded_sph[:, :, :, w - 2 * padding_margin:w - padding_margin]
    rep_padded_sph[:, :, :, h - padding_margin:] = rep_padded_sph[:, :, :, padding_margin:2 * padding_margin]
    return rep_padded_sph


class render_spherical(torch.nn.Module):
    def __init__(self, sph_res=128, z_res=256):
        super().__init__()
        self.sph_res = sph_res
        self.z_res = z_res
        self.gen_grid()
        self.calc_stop_prob = CalcStopProb().apply

    def gen_grid(self):
        res = self.sph_res
        z_res = self.z_res
        pi = np.pi
        phi = np.linspace(0, 180, res * 2 + 1)[1::2]
        theta = np.linspace(0, 360, res + 1)[:-1]
        grid = np.zeros([res, res, 3])
        for idp, p in enumerate(phi):
            for idt, t in enumerate(theta):
                grid[idp, idt, 2] = np.cos((p * pi / 180))
                proj = np.sin((p * pi / 180))
                grid[idp, idt, 0] = proj * np.cos(t * pi / 180)
                grid[idp, idt, 1] = proj * np.sin(t * pi / 180)
        grid = np.reshape(grid * 2, (res, res, 3))
        alpha = np.zeros([1, 1, z_res, 1])
        alpha[0, 0, :, 0] = np.linspace(0, 1, z_res)
        grid = grid[:, :, np.newaxis, :]
        grid = grid * (1 - alpha)
        grid = torch.from_numpy(grid).float()
        depth_weight = torch.linspace(0, 1, self.z_res)
        self.register_buffer('depth_weight', depth_weight)
        self.register_buffer('grid', grid)

    def forward(self, vox):
        grid = self.grid.expand(vox.shape[0], -1, -1, -1, -1)
        vox = vox.permute(0, 1, 4, 3, 2)
        prob_sph = torch.nn.functional.grid_sample(vox, grid)
        prob_sph = torch.clamp(prob_sph, 1e-5, 1 - 1e-5)
        sph_stop_prob = self.calc_stop_prob(prob_sph)
        exp_depth = torch.matmul(sph_stop_prob, self.depth_weight)
        back_groud_prob = torch.prod(1.0 - prob_sph, dim=4)
        back_groud_prob = back_groud_prob * 1.0
        exp_depth = exp_depth + back_groud_prob
        return exp_depth
