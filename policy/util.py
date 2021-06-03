import torch.nn as nn


def init_(module):
    nn.init.normal_(module.weight, mean=0., std=0.1)
    nn.init.constant_(module.bias, val=0.)
    return module
