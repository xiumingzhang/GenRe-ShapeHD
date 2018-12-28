import time
import numpy as np
import torch
from torch.autograd import Variable


def cross_prod(u, v):
    # Cross pruduct between a set of vectors and a vector
    if len(u.size()) == 2:
        i = u[:, 1] * v[2] - u[:, 2] * v[1]
        j = u[:, 2] * v[0] - u[:, 0] * v[2]
        k = u[:, 0] * v[1] - u[:, 1] * v[0]
        return torch.stack((i, j, k), 1)
    elif len(u.size()) == 3:
        i = u[:, :, 1] * v[2] - u[:, :, 2] * v[1]
        j = u[:, :, 2] * v[0] - u[:, :, 0] * v[2]
        k = u[:, :, 0] * v[1] - u[:, :, 1] * v[0]
        return torch.stack((i, j, k), 2)
    raise Exception()


def criterion_single(v, x, x_0, n_0, l, alpha=np.sqrt(2) / 2, beta=1, gamma=1.):
    v = v.view(-1)
    x = x.view(-1, 3)
    n_0 /= torch.sum(n_0 ** 2)

    # Find the voxel which is nearest to x_0
    _, index = torch.min(torch.sum((x - x_0) ** 2, dim=1), dim=0)
    i_0 = index.data.cpu().numpy()[0]

    # loss for (i_0, j_0, k_0)
    loss_1 = (1 - v[i_0]) ** 2

    # loss for others
    d = torch.sum(cross_prod((x - x_0), n_0) ** 2, dim=1) ** 0.5
    mask_1 = (d < alpha * l).float()
    mask_2 = torch.ones(*v.size())
    mask_2[i_0] = 0
    mask_2 = Variable(mask_2.cuda())
    loss_2 = torch.sum((gamma * (1 - d / (alpha * l)) ** beta * v ** 2) * mask_1 * mask_2)

    return loss_1 + loss_2


def criterion(v, x, x_0, n_0, l, alpha=np.sqrt(2) / 2, beta=1, gamma=1.):
    n_sample = x_0.size(0)
    v = v.view(-1)
    x = x.view(-1, 3)
    n_0 /= torch.sum(n_0 ** 2)

    # Find the voxel which is nearest to x_0
    x_repeat = x.view(x.size(0), 1, x.size(1)).repeat(1, n_sample, 1)
    x_sub = x_repeat - x_0
    _, index = torch.min(torch.sum(x_sub ** 2, dim=2), dim=0)
    i_0 = index.data.cpu().numpy()
    
    # loss for (i_0, j_0, k_0)
    loss_1 = Variable(torch.zeros(1).cuda())
    for i in range(n_sample):
        loss_1 += (1 - v[i_0[i]]) ** 2
    
    # loss for others
    d = torch.sum(cross_prod(x_sub, n_0) ** 2, dim=2) ** 0.5
    mask_1 = (d < alpha * l).float()
    mask_2 = torch.ones(v.size(0), n_sample)
    for i in range(n_sample):
        mask_2[i_0[i]][i] = 0
    mask_2 = Variable(mask_2.cuda())
    v_repeat = v.view(v.size(0), 1).repeat(1, n_sample)
    loss_2 = torch.sum((gamma * (1 - d / (alpha * l)) ** beta * v_repeat ** 2) * mask_1 * mask_2)
    return loss_2


if __name__ == '__main__':
    torch.manual_seed(70)
    n_sample = 90
    N = 128
    l = 1.
    v = Variable(torch.rand(N, N, N).cuda(), requires_grad=True)
    x = Variable(torch.rand(N, N, N, 3).cuda())
    x_0 = Variable(torch.rand(n_sample, 3).cuda())
    n_0 = Variable(torch.rand(3).cuda())

    start = time.time()
    
    loss = criterion(v, x, x_0, n_0, l)

    '''
    loss = Variable(torch.zeros(1).cuda())
    for i in range(n_sample):
        loss += criterion_single(v, x, x_0[i], n_0, l)
    '''
    
    loss.backward()
    print(v.grad[0, 0, 0])
    
    end = time.time()
    print(end - start)
    

    u = Variable(torch.rand(N, 3).cuda())
    v = Variable(torch.rand(3).cuda())
    # print(cross_prod(u, v))

    # print(np.cross(u.data.cpu().numpy()[0], v.data.cpu().numpy()))
