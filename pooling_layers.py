# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:55:56 2021

@author: htngu
I use "pooling" in the title very loosely here. What I mean 
is a set of a layers that sort of transform . I don't want to say reduce
dimensions because it's not quite the primary objective - though it does
help.

In order:
+ AttentionHead: attention pooling layer followed by a fully-connected layer
+ Attention Pooling layer
+ GeM pooling layer: generalization of max/avg pooling
+ netVLAD pooling layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

from torchvision import models
from torch.autograd import Variable

########################################################
class AttentionHead(nn.Module):
    '''
    attention-head: consists of an attention-based pooling layer
    followed by a fully-connected layer
    '''
    def __init__(self, c_in, c_out, n_tiles):
        self.n_tiles = n_tiles
        super().__init__()
        self.attention_pool = AttentionPool(c_in, c_in//2)
        self.fc = nn.Linear(c_in, c_out)

    def forward(self, x):

        bn, c = x.shape
        h = x.view(-1, self.n_tiles, c)
        h = self.attention_pool(h)
        h = self.fc(h)
        return h


class AttentionPool(nn.Module):
    '''
    Attention Pooling layer
    implementation by 
    
    '''
    def __init__(self, c_in, d):
        super().__init__()
        self.lin_V = nn.Linear(c_in, d)
        self.lin_w = nn.Linear(d, 1)

    def compute_weights(self, x):
        key = self.lin_V(x)  # b, n, d
        weights = self.lin_w(torch.tanh(key))  # b, n, 1
        weights = torch.softmax(weights, dim=1)
        return weights

    def forward(self, x):
        weights = self.compute_weights(x)
        pooled = torch.matmul(x.transpose(1, 2), weights).squeeze(2)   # b, c, n x b, n, 1 => b, c, 1
        return pooled
########################################################
class GeM(nn.Module):
    '''
    Generalized Mean Pooling: pooling layer that generalizes
    max and average pooling with a trainable parameter p
    
    p:=1 -> recover avg pooling
    p:=inf -> recover max pooling
    '''
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = torch.nn.parameter.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return torch.nn.functional.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + \
               'eps=' + str(self.eps) + ')'

########################################################
class NetVLAD(nn.Module):
    '''
    NetVLAD layer
    implementation by
    based on NetVlad paper
    '''
    def __init__(self, feature_size, max_frames,cluster_size, add_bn=False, truncate=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size / 2 if truncate else feature_size
        self.max_frames = max_frames
        self.cluster_size = cluster_size
        self.batch_norm = nn.BatchNorm1d(cluster_size, eps=1e-3, momentum=0.01)
        self.linear = nn.Linear(self.feature_size, self.cluster_size)
        self.softmax = nn.Softmax(dim=1)
        self.cluster_weights2 = nn.Parameter(torch.FloatTensor(1, self.feature_size,
                                                               self.cluster_size))
        self.add_bn = add_bn
        self.truncate = truncate
        self.first = True
        self.init_parameters()

    def init_parameters(self):
        init.normal_(self.cluster_weights2, std=1 / math.sqrt(self.feature_size))

    def forward(self, reshaped_input):
        random_idx = torch.bernoulli(torch.Tensor([0.5]))
        if self.truncate:
            if self.training == True:
                reshaped_input = reshaped_input[:, :self.feature_size].contiguous() if random_idx[0]==0 else reshaped_input[:, self.feature_size:].contiguous()
            else:
                if self.first == True:
                    reshaped_input = reshaped_input[:, :self.feature_size].contiguous()
                else:
                    reshaped_input = reshaped_input[:, self.feature_size:].contiguous()
        activation = self.linear(reshaped_input)
        if self.add_bn:
            activation = self.batch_norm(activation)
        activation = self.softmax(activation).view([-1, self.max_frames, self.cluster_size])
        a_sum = activation.sum(-2).unsqueeze(1)
        a = torch.mul(a_sum, self.cluster_weights2)
        activation = activation.permute(0, 2, 1).contiguous()
        reshaped_input = reshaped_input.view([-1, self.max_frames, self.feature_size])
        vlad = torch.matmul(activation, reshaped_input).permute(0, 2, 1).contiguous()
        vlad = vlad.sub(a).view([-1, self.cluster_size * self.feature_size])
        if self.training == False:
            self.first = 1 - self.first
        return vlad

