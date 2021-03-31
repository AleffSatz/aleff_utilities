# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:16:57 2021

@author: htngu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, OneCycleLR

class ConcatLR(torch.optim.lr_scheduler._LRScheduler):
    '''
    concat 2 schedulers 
    '''
    def __init__(self, optimizer, scheduler1, scheduler2, total_steps, pct_start=0.5, last_epoch=-1):
        self.scheduler1 = scheduler1
        self.scheduler2 = scheduler2
        self.step_start = float(pct_start * total_steps) - 1
        super(ConcatLR, self).__init__(optimizer, last_epoch)
    
    def step(self):
        if self.last_epoch <= self.step_start:
            self.scheduler1.step()
        else:
            self.scheduler2.step()
        super().step()
        
    def get_lr(self):
        if self.last_epoch <= self.step_start:
            return self.scheduler1.get_lr()
        else:
            return self.scheduler2.get_lr()
        
def FlatAnnealingLR(optimizer, init_lr, annealing_start, total_steps):
    '''
    return the flat annealing lr scheduler
    '''
    flat = LambdaLR(optimizer, lambda x: 1)
    annealing = CosineAnnealingLR(optimizer, total_steps*(1-annealing_start))
    scheduler = ConcatLR(optimizer, flat, annealing, total_steps, annealing_start)
    return scheduler