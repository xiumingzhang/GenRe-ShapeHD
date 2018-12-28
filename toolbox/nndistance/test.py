import torch
from torch.autograd import Variable

try:
    from nndistance.modules.nnd import NNDModule
except ImportError as err:
    raise ImportError('This file should be copied to its parent directory for import to work properly.')

dist = NNDModule()

p1 = torch.rand(1,50,3)*20
p2 = torch.rand(1,50,3)*20
# p1 = p1.int()
# p1.random_(0,2)
# p1 = p1.float()
# p2 = p2.int()
# p2.random_(0,2)
p2 = p2.float()
# print(p1)
# print(p2)

print('cpu')
points1 = Variable(p1, requires_grad=True)
points2 = Variable(p2, requires_grad=True)
dist1, dist2 = dist(points1, points2)
print(dist1, dist2)
loss = torch.sum(dist1)
print(loss)
loss.backward()
print(points1.grad, points2.grad)

print('gpu')
points1_cuda = Variable(p1.cuda(), requires_grad=True)
points2_cuda = Variable(p2.cuda(), requires_grad=True)
dist1_cuda, dist2_cuda = dist(points1_cuda, points2_cuda)
print(dist1_cuda, dist2_cuda)
loss_cuda = torch.sum(dist1_cuda)
print(loss_cuda)
loss_cuda.backward()
print(points1_cuda.grad, points2_cuda.grad)

print('stats:')
print('loss:', loss.data[0], loss_cuda.data[0])
print('loss diff:', loss.data[0] - loss_cuda.data[0])
print('grad diff:', (points1.grad.data.cpu() - points1_cuda.grad.data.cpu()).abs().max(), (points2.grad.data.cpu() - points2_cuda.grad.data.cpu()).abs().max())

from nndistance.functions.nnd import nndistance_score
print('total score:', nndistance_score(points1, points2))
