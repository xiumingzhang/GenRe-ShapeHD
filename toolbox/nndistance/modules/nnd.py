from torch.nn import Module
from nndistance.functions.nnd import nndistance


class NNDModule(Module):
    def forward(self, input1, input2):
        return nndistance(input1, input2)
