#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 13:37:44 2020

My development script for use in data exploration and use in Jupyter notebook
N.B.: Code not cleaned or annotated. 

@author: drmdm
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplleaflet
import os

from numpy.random import seed
from tensorflow import set_random_seed

from keras.layers.core import Dense 
from keras.models import Sequential
from keras import regularizers

import warnings
warnings.filterwarnings("ignore")

datadir="./data/"
filelist=os.listdir(datadir)
file=filelist[5]

def df(datadir, file, mask=True):
    df=pd.read_csv(datadir+file)
    df.columns = [col.strip() for col in df.columns] 
        
    df['mag']=np.sqrt((np.square(df['x']) + np.square(df['y']) + np.square( df['z'])))
    if df['mag'].max() > 15:
        df[['x', 'y', 'z']]=df[['x', 'y', 'z']].div(9.81)
        df['mag']=np.sqrt((np.square(df['x']) + np.square(df['y']) + np.square( df['z'])))
    
    df['a_max']=df[['x', 'y', 'z']].max(axis=1)    
    df['a_min']=df[['x', 'y', 'z']].min(axis=1)  
  
    max_x, max_y, max_z, max_mag =  df['x'].max(), df['y'].max(), df['z'].max(), df['mag'].max()
    min_x, min_y, min_z, min_mag =  df['x'].min(), df['y'].min(), df['z'].min(), df['mag'].min() 
    max_speed=df['speed'].max()
  
    stats={ 
       'max_x'  : max_x,
       'max_y'  : max_y,
       'max_z'  : max_z,
       'max_mag': max_mag,
       'min_x'  : min_x,
       'min_y'  : min_y,
       'min_z'  : min_z,
       'min_mag': min_mag,
       'max_speed': max_speed
       }


    df['Datetime']= pd.to_datetime(df['timestamp'], unit='ms')   
    start=df['Datetime'].min().round('s')
    end=df['Datetime'].max().round('s')
    duration=end-start
    
    t = np.linspace(start.value, end.value, int(duration.seconds+1))
    t = pd.to_datetime(t)
    resample=df
    test=resample.resample('0.5S', on='Datetime').mean().ffill().reset_index()
    df=test.drop(columns=['timestamp', 'height'])
         
    df['t_delta']=(df['Datetime']-df['Datetime'].shift()).apply(lambda x: x.total_seconds())
    df['triptime']=(df['Datetime']-df['Datetime'].iloc[0]).apply(lambda x: x.total_seconds())

    if mask:
        _mask = df['speed'] > 8
        df[['x', 'y', 'z']] = df[['x', 'y', 'z']].apply(lambda x: np.where(_mask, x, 0))

    df['x_diff']=df['x'].diff()
    df['y_diff']=df['y'].diff()
    df['z_diff']=df['z'].diff()
    df['mag_diff']=np.sqrt((np.square(df['x_diff']) + np.square(df['y_diff']) + np.square( df['z_diff'])))
    
    df['jerk_x']=((df['x']-df['x'].shift(1))/df['t_delta'])
    df['jerk_y']=((df['y']-df['y'].shift(1))/df['t_delta'])
    df['jerk_z']=((df['z']-df['z'].shift(1))/df['t_delta'])
    df['jerk_mag']=np.sqrt((np.square(df['jerk_x']) + np.square(df['jerk_y']) + np.square( df['jerk_z'])))
    
    df['v_x']=((np.square(df['x'])/2)-(np.square(df['x'].shift(1)))/2)
    df['v_y']=((np.square(df['y'])/2)-(np.square(df['y'].shift(1)))/2)
    df['v_z']=((np.square(df['z'])/2)-(np.square(df['z'].shift(1)))/2)
    df['v_mag']=np.sqrt((np.square(df['v_x']) + np.square(df['v_y']) + np.square( df['v_z'])))  
    df['v_min']=df[['v_x', 'v_y', 'v_z']].min(axis=1)
    
    max_x, max_y, max_z, max_mag =  df['x'].max(), df['y'].max(), df['z'].max(), df['mag'].max()
    min_x, min_y, min_z, min_mag =  df['x'].min(), df['y'].min(), df['z'].min(), df['mag'].min()  
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
       'min_mag': max_mag,
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
    
    return df, stats, avg_stats


def simple_autoencoder(df, columns):   
    X_train=df[columns]
    X_train.index=df.triptime
    X_train.dropna(inplace=True)
      
    seed(10)
    set_random_seed(10)
    act_func = 'elu'
    
    # Input layer:
    model=Sequential()
    # First hidden layer, connected to input vector X. 
    model.add(Dense(10,activation=act_func,
                    kernel_initializer='glorot_uniform',
                    kernel_regularizer=regularizers.l2(0.0),
                    input_shape=(X_train.shape[1],)))

    model.add(Dense(2,activation=act_func,
                    kernel_initializer='glorot_uniform'))    
    model.add(Dense(10,activation=act_func,
                    kernel_initializer='glorot_uniform'))    
    model.add(Dense(X_train.shape[1],
                    kernel_initializer='glorot_uniform'))    
    model.compile(loss='mse',optimizer='adam')
    

    NUM_EPOCHS=10
    BATCH_SIZE=10  
    
    history=model.fit(np.array(X_train),np.array(X_train),
                  batch_size=BATCH_SIZE, 
                  epochs=NUM_EPOCHS,
                  validation_split=0.05,
                  verbose = 1)
    
    plt.figure()
    plt.plot(history.history['loss'], 'b', label='Training loss')
    plt.plot(history.history['val_loss'],'r', label='Validation loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss, [mse]')
    plt.show()
        
    
    X_pred = model.predict(np.array(X_train))
    X_pred = pd.DataFrame(X_pred, columns=X_train.columns)
    X_pred.index = X_train.index
    
    scored = pd.DataFrame(index=X_train.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
    plt.figure()
    sns.distplot(scored['Loss_mae'],
                 bins = 10, 
                 kde= True,
                 color = 'blue');
    
    X_pred_train = model.predict(np.array(X_train))
    X_pred_train = pd.DataFrame(X_pred_train,columns=X_train.columns)
    X_pred_train.index = X_train.index
    
    scored_train = pd.DataFrame(index=X_train.index)
    scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
    scored_train['Threshold'] = 0.5
    scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
    

    scored_train.plot(logy=True,  figsize = (10,6), color = ['blue','red'])
    return scored_train['Anomaly']

def deep_autoencoder(df, columns):
    
    X_train=df[columns]
    X_train.dropna(inplace=True)
      
    seed(10)
    set_random_seed(10)
    activation = 'elu'
    
    # Input layer:
    model=Sequential()
    # First hidden layer, connected to input vector X. 
    model.add(Dense(16, activation=activation, input_shape=(X_train.shape[1],)))
    model.add(Dense(8, activation=activation))
    model.add(Dense(2, activation=activation))
    
    model.add(Dense(8, activation=activation))
    model.add(Dense(16, activation=activation))
    model.add(Dense(X_train.shape[1], activation='sigmoid'))
       
    model.compile(loss='mse',optimizer='adam'   )

    NUM_EPOCHS=10
    BATCH_SIZE=10  
    
    history=model.fit(np.array(X_train),np.array(X_train),
                  batch_size=BATCH_SIZE, 
                  epochs=NUM_EPOCHS,
                  validation_split=0.05,
                  verbose = 1)
    
    plt.figure()
    plt.plot(history.history['loss'], 'b', label='Training loss')
    plt.plot(history.history['val_loss'], 'r', label='Validation loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss, [mse]')
    plt.show()
        

    X_pred = model.predict(np.array(X_train))
    X_pred = pd.DataFrame(X_pred, columns=X_train.columns)
    X_pred.index = X_train.index
    
    scored = pd.DataFrame(index=X_train.index)
    scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
    plt.figure()
    sns.distplot(scored['Loss_mae'],
                 bins = 10, 
                 kde= True,
                 color = 'blue')    
    
    X_pred_train = model.predict(np.array(X_train))
    X_pred_train = pd.DataFrame(X_pred_train, columns=X_train.columns)
    X_pred_train.index = X_train.index
    
    scored_train = pd.DataFrame(index=X_train.index)
    scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-X_train), axis = 1)
    scored_train['Threshold'] = 1.0
    print(scored_train.shape)
    scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']

    scored_train.plot(logy=True,  figsize = (10,6), color = ['blue','red'])  
    return scored_train['Anomaly']


def dbscan(df):    
    from sklearn.cluster import DBSCAN
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches
    
    X=df[['x','y','z']].as_matrix()
    dbscan = DBSCAN(eps = 1, min_samples = 2)
    
    clst=dbscan.fit_predict(X)
    print("Cluster membership values:\n{}".format(clst))
    
    class_labels=['Noise', 'Cluster 0', 'Cluster 1']
    num_labels = len(class_labels)

    marker_array = ['o', '^', '*']
    color_array = ['#FFFF00', '#00AAFF', '#000000', '#FF00AA']
    cmap_bold = ListedColormap(color_array)
    bnorm = BoundaryNorm(np.arange(0, num_labels + 1, 1), ncolors=num_labels)
    plt.figure()

    plt.scatter(X[:, 0], X[:, 1], s=65, cmap=cmap_bold, norm = bnorm, alpha = 0.40, edgecolor='black', lw = 1)
    plt.title('DBSCAN')

    h = []
    for c in range(0, num_labels):
        h.append(mpatches.Patch(color=color_array[c], label=class_labels[c]))
    plt.legend(handles=h)

    plt.show()
    
def ibm_dbscan(df, p1, p2):
    from sklearn.cluster import DBSCAN
    import sklearn.utils
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    sklearn.utils.check_random_state(1000)
    
    Clus_dataSet = df[[p1, p2]]
    Clus_dataSet = np.nan_to_num(Clus_dataSet)
    Clus_dataSet = StandardScaler().fit_transform(Clus_dataSet)


    # Compute DBSCAN
    db = DBSCAN(eps=1.5, min_samples=15).fit(Clus_dataSet)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    df["Clus_Db"]=labels
    
    realClusterNum=len(set(labels)) - (1 if -1 in labels else 0)
    clusterNum = len(set(labels)) 
    
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print('ClusterNum: %d' % clusterNum)
    
    ids=np.where(db.labels_==-1)      
    crash=df[['Datetime', 'speed', 'mag', 'jerk_mag', 'v_min']].loc[ids[0]]     
    print(crash)
    
    plt.figure() 
    colors = plt.get_cmap('jet')(np.linspace(0.0, 1.0, clusterNum))
    for clust_number in set(labels):
        c=(([0.4,0.4,0.4]) if clust_number == -1 else colors[np.int(clust_number)])
        clust_set = df[df.Clus_Db == clust_number]                  
        plt.scatter(clust_set[p1], clust_set[p2], color=c,  marker='o', s= 20, alpha = 0.85)
        if clust_number != -1:
            cenx=np.mean(clust_set[p1]) 
            ceny=np.mean(clust_set[p2]) 
            plt.text(cenx,ceny,str(clust_number), fontsize=25, color='red',)
    plt.show()

def rolling_avg(df):
    x_avg = df['x'].rolling(window=10).mean()
    plt.plot(x_avg, linewidth=0.5)
    plt.plot(df['x'], alpha=0.5, linewidth=0.5)
    
def fft(df):
    signal = df['y']
    fourier = np.fft.fft(signal)/len(signal)
    timestep = 0.1
    freq = np.fft.fftfreq(len(signal), d=timestep)
    plt.plot(freq, np.abs(fourier))
    
    pos_mask = np.where(freq > 0)
    freqs = freq[pos_mask]
    peak_freq = freqs[np.abs(fourier)[pos_mask].argmax()]
    
    high_freq_fft = fourier.copy()
    high_freq_fft[np.abs(freq) > peak_freq] = 0
    filtered_sig = np.fft.ifft(high_freq_fft)
    plt.plot(df['triptime'], filtered_sig)
    
    """
    # Frequency domain representation
    timestep = 0.1
    fourierTransform = np.fft.fft(signal)/len(signal)           # Normalize amplitude
    fourierTransform = fourierTransform[range(int(len(signal)/2))] # Exclude sampling frequency
         
    tpCount     = len(signal)
    values      = np.arange(int(tpCount/2))
    timePeriod  = tpCount/timestep
    frequencies = values/timePeriod
    
    # Frequency domain representation
    plt.plot(frequencies, abs(fourierTransform))
    """
    
def leaflet_plot(df):
    plt.figure(figsize=(8,8))       
    plt.scatter(df['lon'], df['lat'], c=df['speed'], s=30, cmap="YlOrRd")
    plt.scatter(df['lon'].iloc[0], df['lat'].iloc[0], s=40, color='black', marker='>')
    return mplleaflet.show()


def acc_plot(df, columns, title):
    fig, ax1 = plt.subplots(figsize=(8,8))    
    ax1.scatter(df['triptime'], df[columns[0]], c='blue', s=1, alpha=0.5)
    ax1.scatter(df['triptime'], df[columns[1]], c='red', s=1, alpha=0.5)
    ax1.scatter(df['triptime'], df[columns[2]], c='green', s=1, alpha=0.5)
    ax1.set_ylabel(title)
    ax1.set_xlabel('Journey Time (s)')
    ax2 = ax1.twinx()
    ax2.plot(df['triptime'], df['speed'], c='black', linewidth=0.75, alpha=0.5)
    ax2.set_ylabel('Speed (mph)')
    plt.title('%s Plot' % title)
    return plt.show()

def acc_cluster(df, columns, title):
    plt.figure(figsize=(8,8)) 
    plt.scatter(df[columns[0]], df[columns[1]], c='blue', s=1, alpha=0.5)    
    plt.scatter(df[columns[0]], df[columns[2]], c='green', s=1, alpha=0.5)  
    plt.scatter(df[columns[1]], df[columns[2]], c='red', s=1, alpha=0.5) 
    plt.title('%s Cluster Plot' % title)
    plt.legend([(columns[0], columns[1]), (columns[0], columns[2]), (columns[1], columns[2])])
    plt.xlabel(title)
    plt.ylabel(title)
    return plt.show()