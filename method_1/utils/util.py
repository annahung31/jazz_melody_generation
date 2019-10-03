import os
import time
import torch
import ipdb
import numpy as np
from torch import nn

import json
from pathlib import Path
from datetime import datetime
from collections import OrderedDict




def one_hot(event, event_dim):
    event_one_hot = torch.ones((event.shape[0],event_dim))
    for i in range(event.shape[0]):
        idx = event[i]
        event_one_hot[i][idx] = int(1)
    return event_one_hot.unsqueeze(1)

def slerp(a, b, steps):
    aa =  np.squeeze(a/np.linalg.norm(a))
    bb =  np.squeeze(b/np.linalg.norm(b))
    ttt = np.sum(aa*bb)
    omega = np.arccos(ttt)
    so = np.sin(omega)
    step_deg = 1 / (steps+1)
    step_list = []

    for idx in range(1, steps+1):
       t = step_deg*idx
       tmp = np.sin((1.0-t)*omega) / so * a + np.sin(t*omega)/so * b
       step_list.append(tmp)
    return step_list



def make_dir():
    dir_name = time.strftime("%Y-%m-%d_%H", time.localtime())
    if not os.path.exists('/home/annahung/project/anna_jam/presets/'+ dir_name):
        os.makedirs('/home/annahung/project/anna_jam/presets/'+ dir_name)
    return '/home/annahung/project/anna_jam/presets/'+ dir_name + '/'


def make_loss_dir():
    dir_name = time.strftime("%Y-%m-%d_%H", time.localtime())
    if not os.path.exists('/home/annahung/project/anna_jam/loss/'+ dir_name):
        os.makedirs('/home/annahung/project/anna_jam/loss/'+ dir_name)
    return '/home/annahung/project/anna_jam/loss/'+ dir_name + '/'




def pitch_type(sample):
    m = torch.argmax(sample[:,:,:-16].reshape(sample.shape[0], sample.shape[1]*16, 49), 2)
    total_pitch_type = 0
    for i in range(m.shape[0]):
        pitch_type = len(set(m[i].tolist()))
        total_pitch_type += pitch_type

    ave_pitch_type = total_pitch_type / sample.shape[0]
    return ave_pitch_type

def pitch_histogram():
    cy_pitch_his = [0.000689075 , 0.001366775 , 0.0004455  , 0.0043005  , 0.002234025  ,0.0001037  ,0.00810475 ,0.000872025, 0.01131225 ,0.00058815 ,0.0147975 ,0.02846 ,0.000208975 ,0.03618 ,0.0059555 ,0.048635 ,0.01960725  ,0.00297215  ,0.05667 ,0.00517525 , 0.0553425  ,0.00644475 ,0.0558925 , 0.09027 ,0.00135875 ,0.0800375  ,0.01126475 ,0.0859925 , 0.0348375  ,0.0023509 ,0.07095 ,0.00342525 , 0.047415 ,0.005838575, 0.0478875 ,  0.05793 ,0.00038245 , 0.0337975  , 0.00395875  ,0.0240975 ,  0.0079635  , 0.000311 ,0.00940525 ,0.000504875, 0.005625, 0.000446, 0.002478525, 0.001769525]
    cy_pitch_his = torch.FloatTensor(cy_pitch_his)
    return cy_pitch_his



def lrelu(x, leak=0.2):
    z = torch.mul(x,leak)
    return torch.max(x, z)


def batch_norm_1d(x):
    x_shape = x.shape[1]
    batch_nor = nn.BatchNorm1d(x_shape, eps=1e-05, momentum=0.9, affine=True)
    batch_nor = batch_nor.cuda()

    output = batch_nor(x)
    return output



def batch_norm_2d(x):
    x_shape = x.shape[1]
    batch_nor = nn.BatchNorm2d(x_shape, eps=1e-05, momentum=0.9, affine=True)
    batch_nor = batch_nor.cuda()
    output = batch_nor(x)
    return output


def reduce_mean(x):
    output = torch.mean(x,0, keepdim = False)
    output = torch.mean(output,-1, keepdim = False)
    return output

def reduce_mean_0(x):
    output = torch.mean(x,0, keepdim = False)
    return output

def sigmoid_cross_entropy_with_logits(inputs,labels):
    loss = nn.BCEWithLogitsLoss()
    output = loss(inputs, labels)
    return output



def l2_loss(x,y):
    loss_ = nn.MSELoss(reduction='sum')
    l2_loss_ = loss_(x, y)/2
    return l2_loss_




def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


def torch_summarize(model, show_weights=False, show_parameters=False):
    """Summarizes torch model by showing trainable parameters and weights."""
    from torch.nn.modules.module import _addindent

    tmpstr = model.__class__.__name__ + ' (\n'
    for key, module in model._modules.items():
        # if it contains layers let call it recursively to get params and weights
        if type(module) in [
            torch.nn.modules.container.Container,
            torch.nn.modules.container.Sequential
        ]:
            modstr = torch_summarize(module)
        else:
            modstr = module.__repr__()
        modstr = _addindent(modstr, 2)

        params = sum([np.prod(p.size()) for p in module.parameters()])
        weights = tuple([tuple(p.size()) for p in module.parameters()])

        tmpstr += '  (' + key + '): ' + modstr
        if show_weights:
            tmpstr += ', weights={}'.format(weights)
        if show_parameters:
            tmpstr += ', parameters={}'.format(params)
        tmpstr += '\n'

    tmpstr = tmpstr + ')'
    return tmpstr
