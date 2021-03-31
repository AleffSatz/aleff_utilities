# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:46:41 2021

@author: htngu

utility functions for deep learning in pytorch

in order:
    + 
"""

import os 
import logging
import sys
import json 
import numpy as np 
import random
import torch 
from torch.nn import DataParallel
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

def save_model(model, path):
    '''
    save model's state_dict
    '''
    if isinstance(model, DataParallel):
        model = model.module

    with open(path, "wb") as fout:
        torch.save(model.state_dict(), fout)
        
def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)
        
def seed_torch(seed=42):
    '''
    set training random seed
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def accuracy(preds,labels,num_classes):
    '''
    raw accuracy
    '''
    classes = range(num_classes)
    metric = {}
    for c in classes:
        pred = np.equal(c,preds)
        label = np.equal(c,labels)
        hit = np.logical_and(pred,label)
        pos = np.sum(label.astype(int))
        hit = np.sum(hit.astype(int))
        if pos==0:
            acc = 0.
        else:
            acc = hit/pos
        metric[c] = acc
    return metric