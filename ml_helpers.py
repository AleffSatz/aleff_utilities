#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hieunguyen

This file includes the following ml helpers:
+ train/test data domain shift checking with random forest
+ computation of generalized ipr
+ focal loss function for gradient boosting models
+ computation of feature skewness
    
"""
import os
import pandas as pd
import numpy as np
import glob
import seaborn as sns
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import pickle
from tempfile import mkdtemp



###############################  check domain shift using random forest       #######################
def check_domain_shift(train_data, test_data
                       , remove_cols=[]
                       , encode_category=True
                      ):
    '''
    fit a train/test classifier to detect possible domain shift
    
    return a df of feature importance
    '''
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier
    
    le = LabelEncoder()
    train_data['dataset_label'] = 0
    test_data['dataset_label'] = 1
    dataset = pd.concat([train_data, test_data], axis=0)
    remaining_cols = [col for col in dataset.columns.values if col not in remove_cols + ['dataset_label']]
    categorical_cols = [col for col in dataset.columns.values if dataset[col].dtypes is np.dtype('O')]
    categorical_cols = [col for col in categorical_cols if col not in remove_cols]
    if encode_category:
        for col in categorical_cols:
            print(col)
            dataset[col] = le.fit_transform(dataset[col].fillna(''))

    domain_rf = RandomForestClassifier(verbose=1, n_jobs=30, n_estimators = 30)
    domain_rf.fit(dataset[remaining_cols].fillna(-1)
              , dataset['dataset_label']
             )
    
    feature_importance = pd.DataFrame({'feature': remaining_cols
                                  , 'importance': domain_rf.feature_importances_
                                  })
    return feature_importance

#########################  compute generlized ipr aka effective number of species  #########################
from numpy.linalg import norm
def compute_ipr_generalized(x_values, q=2):
    '''
    compute the generalized effective number of species
    
    the higher q is, the more weight attributed to the most abundant types
    when q=2, we recover the usual ipr
    '''
    epsilon = 1e-10
    x_values = np.array(x_values)
    p_i = x_values/(np.sum(x_values) + epsilon)
    norm_q = norm(p_i, ord=q)
    ipr_generalized = np.power(norm_q, q/(1-q)) if norm_q != 0. else 0.
    return ipr_generalized

#######################  custom loss for gradient boosting  ########################################

def _sigmoid(x):
    '''
    stable sigmoid python implementation
    '''
    return np.where(x.astype(float) >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))

def focal_loss_lgb_sklearn(y_true, y_pred):
    '''
    implement focal loss function for lgb
    return gradient and hessian
    '''
    alpha=0.25; gamma=1.5
    a, g = alpha, gamma
    #y_true = dtrain.get_label()
    yt = (2. * y_true - 1.)
    at = y_true * a + (1-a)*(1-y_true) 
    #at = 1.
    p = _sigmoid(y_pred)
    pt = y_true * p + (1. - p)*(1-y_true)
    dpt = (2. * y_true - 1) * (p*(1-p))
    dp = p*(1.-p)
    gradient = yt * at * np.power(1.-pt, g) * (g * pt * np.log(pt) + pt - 1)
    hessian_multiplier = (- pt * np.log(pt) * (np.power(g,2) + g)
                          + g * np.log(pt) - (2.*g + 1) * pt
                          + 2. * g + 1.
                         )
    hessian = yt * at * np.power(1.-pt, g-1) * dpt * hessian_multiplier
    
    return gradient, hessian

def cross_entropy_loss_lgb(y_true, y_pred):
    '''
    the usual cross-entropy loss
    
    not technically a custom loss, more a tester to see if custom loss is functioning
    '''

    p = 1./(1. + np.exp(-y_pred))
    grad = p - y_true
    hess = p * (1. - p)
    return grad, hess

'''
class focal_loss_torch(nn.Module):
    def __init__(self, alpha=1., gamma=1.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()

'''
############################  compute feature skewness   ############################3
def compute_feature_skewness(temp_df, numerical_cols, absolute_value=True):
    '''
    input: data frame, list of numerical features
    return a df for feature (absolute) skewness to guide boxcox transform
    '''
    from scipy.stats import skew
    #skew_feats = temp_df[numerical_cols].apply(lambda x: np.abs(skew(x)), axis=0).sort_values(ascending=False)
    skew_feats = (temp_df[numerical_cols]
                  .skew(axis=0)
                  .rename('skewness').reset_index()
                  .rename(columns={'index':'feature'})
                 )
    if (absolute_value):
        skew_feats['skewness'] = np.abs(skew_feats['skewness'])
    skew_feats.sort_values('skewness', ascending=False)
    
    return skew_feats