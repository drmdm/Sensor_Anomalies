#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 16:45:57 2020

Module to support crash event prediction using phone accelerometer data.

@author: drmdm
"""
import numpy as np
import pandas as pd

from numpy.random import seed
from tensorflow import set_random_seed
from keras.layers.core import Dense 
from keras.models import Sequential
from keras import regularizers

import warnings
warnings.filterwarnings("ignore")

def df(datadir, file):
    """
    Load the journey data and process it ready for modelling.

    Parameters
    ----------
    datadir : str
        The folder path to where the journey data is stored
        
    file : str
        Filename of the file to open
        

    Returns
    -------
    df : Pandas DataFrame
        DataFrame contain the processed data
        
    avg_stats : Pandas DataFrame
        DataFrame containing the statistics of df
    """
    df=pd.read_csv(datadir+file)
    
    #Remove spaces from the columns names
    df.columns = [col.strip() for col in df.columns] 

    #Find acceleration magnitude
    df['mag']=np.sqrt((np.square(df['x']) + np.square(df['y']) + np.square( df['z'])))
    
    #Check if the acceleration is in G or ms^-2. Convert to G
    if df['mag'].max() > 15:
        df[['x', 'y', 'z']]=df[['x', 'y', 'z']].div(9.81)
        df['mag']=np.sqrt((np.square(df['x']) + np.square(df['y']) + np.square( df['z'])))
    
    #Find max/min acceleration from accelerometer
    df['a_max']=df[['x', 'y', 'z']].max(axis=1)    
    df['a_min']=df[['x', 'y', 'z']].min(axis=1)  
  
    #Convert unix timestamps to Datetime
    df['Datetime']= pd.to_datetime(df['timestamp'], unit='ms')   
    
    #Journey details used to create equal time periods
    start=df['Datetime'].min().round('s')
    end=df['Datetime'].max().round('s')
    duration=end-start
    
    #Create equal time periods (0.5 second steps) and resample the time series
    t = np.linspace(start.value, end.value, int(duration.seconds+1))
    t = pd.to_datetime(t)
    resample=df
    test=resample.resample('0.5S', on='Datetime').mean().ffill().reset_index()
    df=test.drop(columns=['timestamp', 'height'])
    
    #Add time delta for calculations and trip time     
    df['t_delta']=(df['Datetime']-df['Datetime'].shift()).apply(lambda x: x.total_seconds())
    df['triptime']=(df['Datetime']-df['Datetime'].iloc[0]).apply(lambda x: x.total_seconds())

    #Calculate jerk (derivative of acceleration)
    df['jerk_x']=((df['x']-df['x'].shift(1))/df['t_delta'])
    df['jerk_y']=((df['y']-df['y'].shift(1))/df['t_delta'])
    df['jerk_z']=((df['z']-df['z'].shift(1))/df['t_delta'])
    df['jerk_mag']=np.sqrt((np.square(df['jerk_x']) + np.square(df['jerk_y']) + np.square( df['jerk_z'])))
    
    #Calculate change in velocity (integral of a.dt evaluated between each timestep)
    df['v_x']=((np.square(df['x'])/2)-(np.square(df['x'].shift(1)))/2)
    df['v_y']=((np.square(df['y'])/2)-(np.square(df['y'].shift(1)))/2)
    df['v_z']=((np.square(df['z'])/2)-(np.square(df['z'].shift(1)))/2)
    df['v_mag']=np.sqrt((np.square(df['v_x']) + np.square(df['v_y']) + np.square( df['v_z'])))  
    df['v_min']=df[['v_x', 'v_y', 'v_z']].min(axis=1)
    
    #Calculate some summary statistics
    max_x, max_y, max_z, max_mag =  df['x'].max(), df['y'].max(), df['z'].max(), df['mag'].max()
    min_x, min_y, min_z =  df['x'].min(), df['y'].min(), df['z'].min()
    max_speed=df['speed'].max()
    min_v=df['v_min'].min()
    a_max=df['a_max'].max()
    a_min=df['a_min'].min()
    
    max_jerk_x, max_jerk_y, max_jerk_z, max_jerk_mag =  df['jerk_x'].max(), df['jerk_y'].max(), df['jerk_z'].max(), df['jerk_mag'].max()
    min_jerk_x, min_jerk_y, min_jerk_z, min_jerk_mag =  df['jerk_x'].min(), df['jerk_y'].min(), df['jerk_z'].min(), df['jerk_mag'].min()          
    avg_stats={
        
       'max_x'  : max_x,
       'max_y'  : max_y,
       'max_z'  : max_z,
       'max_mag': max_mag,
       'min_x'  : min_x,
       'min_y'  : min_y,
       'min_z'  : min_z,
       'max_jerk_x': max_jerk_x,
       'max_jerk_y': max_jerk_y,
       'max_jerk_z': max_jerk_z,
       'max_jerk_mag': max_jerk_mag,
       'min_jerk_x': min_jerk_x,
       'min_jerk_y': min_jerk_y,
       'min_jerk_z': min_jerk_z,
       'min_jerk_mag': min_jerk_mag,
       'max_speed': max_speed,
       'min_v' : min_v,
       'a_max': a_max,       
       'a_min': a_min
       }
    
    return df, avg_stats


def simple_autoencoder(df, columns):
    """
    The Autoencoder is a Keras based neural network which encodes the data, 
    compresses it, then tries to recreate the data. Anomalies are measured based
    on the difference between the true data and the recreation.

    Parameters
    ----------
    df : Pandas DataFrame
        The dataframe created by df()
        DataFrame containing the processed journey information
        
    columns : list
        a list of columns from the df on which to train the autoencoder

    Returns
    -------
    anomaly: Pandas DataFrame
        A DataFrame containing potential collision info.
        use the index to join to the original df. 
        'Anomaly' and 'Loss Mae' values processed using postprocess()
    """
    
    #Create training date from the chosen columns
    X_train=df[columns]
    X_train.index=df.index
    X_train.dropna(inplace=True)
      
    seed(10)
    set_random_seed(10)
    act = 'elu'
    ki  = 'glorot_uniform'
    
    #Set up the model
    model=Sequential()
    model.add(Dense(10,activation=act,
                    kernel_initializer=ki,
                    kernel_regularizer=regularizers.l2(0.0),
                    input_shape=(X_train.shape[1],)))
    
    model.add(Dense(2, activation=act, kernel_initializer=ki)) 
    model.add(Dense(10, activation=act, kernel_initializer=ki))
    model.add(Dense(X_train.shape[1], kernel_initializer=ki)) 
    model.compile(loss='mse',optimizer='adam')
    
    NUM_EPOCHS=10
    BATCH_SIZE=10  
    
    #Fit the model
    history=model.fit(np.array(X_train),np.array(X_train),
                  batch_size=BATCH_SIZE, 
                  epochs=NUM_EPOCHS,
                  validation_split=0.05,
                  verbose=0)

    #Create prediciton (attempt to recreate the data)
    X_pred_train = model.predict(np.array(X_train))
    X_pred_train = pd.DataFrame(X_pred_train,columns=X_train.columns)
    X_pred_train.index = X_train.index
    
    #Calculate the difference between the recreated data and original data.
    #Set a threshold on the difference to detect anomalies.
    scored_train = pd.DataFrame(index=X_train.index)
    scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
    scored_train['Threshold'] = 0.5
    scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
    
    anomaly=scored_train[['Anomaly', 'Loss_mae']]
    return anomaly

def postprocess(file, anomaly, df):   
    """
    Postprocess the autoencoder results to predict a crash event, severity and 
    confidence.    

    Parameters
    ----------
    file : str
        The filename being processed
        
    anomaly : Pandas DataFrame
        DataFrame containing anomaly data from the simple_autoencoder()
        
    df : Pandas DataFrame
        The dataframe created by df()
        DataFrame containing the processed journey information

    Returns
    -------
    result : dict
        A dictionary containing information about potential crash events.

    """
    
    #Find the anomalies
    ids=np.where(anomaly['Anomaly']==True)
    crash=anomaly.iloc[ids]
    
    #If anomalies are detected join the data frame to get journey details.
    if len(crash) > 0:
        full=crash.join(df, how='left')
        full=full[['speed', 'Loss_mae', 'jerk_mag', 'v_min', 'Datetime']]
        
        #Filter any anomalies that occur under 5mph
        #There tends to be a lot of noise when the vehicle is stationary or at low speed
        filt=np.where(full['speed'] > 5)
        full=full.iloc[filt]
        
        #If anomalies are present and speed > 5mph assign values
        if len(full) > 0:  
            maxloss=full['Loss_mae'].idxmax()
            maxloss_val=full['Loss_mae'].loc[maxloss]            
            speed=full['speed'].loc[maxloss]
            jerk=full['jerk_mag'].loc[maxloss]
            dv=full['v_min'].loc[maxloss]
            time=full['Datetime'].loc[maxloss]
            
            #Severity criteria
            if speed > 20:
                severity = 'Serious'
            elif speed > 14:
                severity = 'Moderate'    
            else:
                severity = 'Minor'
                
            #Confidence criteria
            if maxloss_val > 1.5:
                confidence = 'High'
            elif maxloss_val > 0.7:
                confidence = 'Medium'   
            else:
                confidence = 'Low'       
        
            out = {'File': file, 
                   'Crash': 1, 
                   'Time': time, 
                   'Severity': severity, 
                   'Confidence': confidence, 
                   'Collision Speed': speed,
                   'Max Loss': maxloss_val} 
        
        #Return crash=False if speed < 5mph
        else:
            out = {'File': file, 
                   'Crash': 0, 
                   'Time': np.nan, 
                   'Severity': np.nan, 
                   'Confidence': np.nan, 
                   'Collision Speed': np.nan,
                   'Max Loss': np.nan }
            
    #Return crash=False if no anomalies      
    else:
        out = {'File': file, 
               'Crash': 0, 
               'Time': np.nan, 
               'Severity': np.nan, 
               'Confidence': np.nan, 
               'Collision Speed': np.nan,
               'Max Loss': np.nan } 
        
    result=pd.DataFrame([out], columns=out.keys())
    return result 