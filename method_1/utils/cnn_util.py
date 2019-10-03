

import numpy as np
import torch
from torch import nn


def batch_norm_2d(x):
    x_shape = x.shape[1]
    batch_nor = nn.BatchNorm2d(x_shape, eps=1e-05, momentum=0.9, affine=True)
    batch_nor = batch_nor.cuda()
    output = batch_nor(x)
    return output


def batch_norm_2d_cpu(x):
    # x_shape = x.shape[1]
    # batch_nor = nn.BatchNorm2d(x_shape, eps=1e-05, momentum=0.9, affine=True)
    # batch_nor = batch_nor
    # output = batch_nor(x)
    output = x
    return output


def sigmoid_cross_entropy_with_logits(inputs,labels):
    loss = nn.BCEWithLogitsLoss()
    output = loss(inputs, labels)
    return output


def l2_loss(x,y):
    loss_ = nn.MSELoss(reduction='sum')
    l2_loss_ = loss_(x, y)/2
    return l2_loss_



def lrelu(x, leak=0.02):
    z = torch.mul(x,leak)
    return torch.max(x, z)
