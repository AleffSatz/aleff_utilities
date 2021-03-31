# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 10:44:36 2021

@author: htngu

In order:
    + mish activation function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


############ MISH ###############################
#### implementation from iafoss
### github/iafoss/panda/mish.py

### except this mish function which is from the original implementation
### 
@torch.jit.script
def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))

class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        #return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))
        return mish(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    '''
    swap all ReLU layers with Mish 
    '''
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)
            
#################################################################