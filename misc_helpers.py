#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 16:17:24 2019

@author: hieunguyen
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

############################################################################
def set_fig_size(x,y):
    '''
    set size for figure, should be before plotting
    '''
    plt.rcParams['figure.figsize'] = [x, y]
    sns.set(font_scale=2.)
    
def load_json(path, **kwargs):
    '''
    load json file
    '''
    import json
    with open(path,'r') as f:
        json_file = json.load(path, **kwargs)
        f.close()
        
    return json_file

def remove_non_csv(files_list, extension='csv'):
    '''
    remove files not belonging in an extension, default is csv
    '''
    files = [file for file in files_list if file.split('.')[-1] == extension]
    if (len(files) == 0):
        print('There seems to be no relevant files. Please check.')
    return files

def get_box_update_datetime(filename):
    '''
    Input: file name (of box_update)
    Output: date of the file
    '''
    file_date = filename.split('_')[-1]
    file_date = file_date.split('.')[0]
    file_date = datetime.strptime(file_date, '%Y-%m-%d')
    return file_date

def remove_sharp_str(x):
    '''
    Remove encoded phone number from the Note column
    '''
    str_parts = x.split('#')
    if len(str_parts) == 1:
        result = str_parts[0]
    else:
        result = ''.join([str_parts[0], str_parts[-1]])
    return result

def filter_dataframe(temp_df, col, value
                     , positive=True
                    ):
    '''
    filter col of temp_df by value
    '''
    if type(value) is list:
        temp_df = temp_df[temp_df[col].isin(value)].copy()
    else:
        temp_df = temp_df[temp_df[col] == value].copy()
    return temp_df
        
def custom_round(x, base=5):
    return base * round(float(x)/base)

def get_duplicates_entries(temp_df, col='Contract'):
    '''
    return dataframe of duplicated entries
    for non-duplicates, just use drop_duplicates
    
    credit: DSM's answer for the question found here
    stackoverflow.com/questions/14657241/how-do-i-get-a-list-of-all-the-duplicate-items-using-pandas-in-python
    '''
    return pd.concat(g for _, g in temp_df.groupby(col) if len(g) > 1)

def get_month_diff(start_month, end_month, absolute=False):
    '''
    return relative (absolute) difference in months 
    '''
    try:
        month_diff = (end_month.year - start_month.year) * 12 + (end_month.month - start_month.month)
        if absolute: month_diff = abs(month_dif)
        return month_diff
    except Exception as error:
        print(error)
        print(start_month)
        print(end_month)
        
def get_days_of_same_month(today):
    '''
    return all days of the same month as today
    '''
    first_day = today - relativedelta(days=today.day - 1)
    temp_days = [first_day + relativedelta(days=n) for n in range(31)]
    temp_days = [day for day in temp_days if day < first_day + relativedelta(months=1)]
    return temp_days

def get_first_day_of_same_month(today):
    '''
    return the first date of the same month as today
    '''
    first_day = today - relativedelta(days=today.day - 1)
    return first_day
    
        
def parallelization(df_source, func
                   , split_by_col=True
                   , col=''
                   , N_CPU=10
                   ):
    """
    split the dataframe into partitions
    then perform parallel computation on the partitions
    """
    import multiprocessing
    if (split_by_col):
        contract_list = df_source[col].unique().tolist()
        contract_split = np.array_split(contract_list, N_CPU)
        df_split = [df_source[df_source.Contract.isin(split)] for split in contract_split]
        pool = multiprocessing.Pool(N_CPU)
        df_output = pd.concat(pool.map(func, df_split))
        df_output = df_output.reset_index(drop=True)
        pool.close()
        pool.join()
        return df_output
