#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:45:57 2020

Prediciton script for crash event prediction using phone accelerometer data.

@author: drmdm
"""

import pandas as pd
import os
import crash_detector as cd
import time

import warnings
warnings.filterwarnings("ignore")

#Enter the directory containing your .csv files prediciton
#Place only your journey files in this folder
datadir="./data/"
filelist=os.listdir(datadir)

#If you want to check a single file enter either the filename or the filelist location
file=filelist[5]

def run_single_file(file):
    df, avg_stats = cd.df(datadir, file)
    anomaly=cd.simple_autoencoder(df, columns=['mag', 'jerk_mag', 'v_min'])
    result=cd.postprocess(file, anomaly, df)
    print(result)
   
def predict_all(filelist):
    start=time.time()
    avg_stats_df=pd.DataFrame()
    results_df=pd.DataFrame()   
    
    for i, files in enumerate(filelist):
        this_run_start=time.time()
        print('Reading file %s: %s' % (str(i), files))     
        df, avg_stats = cd.df(datadir, files)
        anomaly=cd.simple_autoencoder(df, columns=['mag', 'jerk_mag', 'v_min'])
        result=cd.postprocess(files, anomaly, df)     
        results_df=results_df.append(result)   
        stats=pd.DataFrame([avg_stats], columns=avg_stats.keys())
        avg_stats_df=avg_stats_df.append(stats)
        
        this_run_end=time.time()
        print(' File %s took %.2f seconds to process\n' % (str(i), this_run_end-this_run_start))
        
    results_df.reset_index(inplace=True, drop=True)
    avg_stats_df.reset_index(inplace=True, drop=True)
    end=time.time()
    runtime=end-start
    print('Program took %s seconds to run' % str(runtime))
    
    return results_df, avg_stats_df

results_df, avg_stats_df = predict_all(filelist)    
results_df.to_csv('./results.csv')