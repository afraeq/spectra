#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 21:23:07 2018

@author: afranio
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from spectra import spectra
from spectra import spectra_dynamics

#%% test 1 -  variance spectra

t1 = np.arange(1,101)
sinal1 = np.sin(t1/4)

# creating example 1 instance
test1 = spectra(sinal1)

# calculating example 1 spectrum, with sliding windows
test1.calc_Var_Spectra('sliding')

# calculating example 1 spectrum, with independent windows
test1.calc_Var_Spectra('independent')

fig, ax = plt.subplots(1,2,figsize=(12,5))

ax[0].plot(t1,sinal1)
test1.plot_Var_Spectra(ax=ax[1])

ax[0].set_xlabel('Time')

ax[0].set_title('Simulated periodic signal');
ax[1].set_title('Variance spectra');
fig.suptitle('Test 1');

#%% test 2 - covariance/correlation spectra

t2 = np.arange(1,301)
sinal2_1 = t2 +100*np.random.rand(t2.size)
sinal2_2 = 3*t2 +100*np.random.rand(t2.size)

test2 = spectra(np.array([sinal2_1,sinal2_2]).T)

test2.calc_Cov_Spectra('sliding',jump_WS=5)
test2.calc_Cov_Spectra('independent',jump_WS=5)

fig, ax = plt.subplots(1,3,figsize=(12,5))

fig.suptitle('Test 2')

ax[0].plot(t2,sinal2_1)
ax[0].set_prop_cycle(None)
ax[0].plot(t2,sinal2_2)

test2.plot_Cov_Spectra(0,1,ax=ax[1],corr_or_cov='cov')
test2.plot_Cov_Spectra(0,1,ax=ax[2],corr_or_cov='corr')

ax[0].set_xlabel('Time')
ax[0].set_title('Simulated signals')
ax[1].set_title('Covariance spectra')
ax[2].set_title('Correlation spectra')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#%% test 3 - latent variance spectra

n=100

t3=np.arange(50)
sinais3 = [j*t3 +np.random.randn(t3.size)*n for j in range(1,11)]

test3 = spectra(np.array(sinais3).T)

test3.calc_Lat_Var_Spectra('sliding')
test3.calc_Lat_Var_Spectra('independent')

fig, ax = plt.subplots(1,2,figsize=(12,5))

fig.suptitle('Test 3')

[ax[0].plot(t3,sinais3[i]) for i in range(len(sinais3))]

ax[0].set_title('Simulated signals')
ax[0].set_xlabel('t')

test3.plot_Lat_Var_Spectra(ax=ax[1])

ax[1].set_title('Latent variance spectra')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#%% test 4 - confidence regions - variance spectra

fig, ax = plt.subplots(2,2,figsize=(12,5))

t4 = np.arange(1,101)

sinal4_train = np.sin(t1/4)
ex4_train = spectra(sinal4_train)
ex4_train.calc_Var_Spectra(percentiles=[25,50,75],jump_WS=2)

sinal4_test = sinal4_train+np.random.rand(sinal4_train.size)
ex4_test = spectra(sinal4_test)
ex4_test.calc_Var_Spectra(percentiles=[25,50,75],jump_WS=2)

ax[0,0].plot(t4,sinal4_train)
ax[1,0].plot(t4,sinal4_test)

[ax[i,0].set_title('Simulated signals') for i in [0,1]]
[ax[i,0].set_xlabel('t') for i in [0,1]]

ex4_train.plot_Var_Spectra(percentile = 50,
                           conf_region=[25,75],
                           ax = ax[0,1])

ex4_test.plot_Var_Spectra(percentile = 50, ax = ax[1,1])

ax[1,1].set_prop_cycle(None)

ax[1,1].fill_between(np.arange(2,ex4_train.spctr_size['sliding'],
                               ex4_train.jump_WS_var['sliding']),
                     ex4_train.lim_inf_var,
                     ex4_train.lim_sup_var,
                     alpha=0.2);  

ax[0,1].set_title('Variance spectra - training')
ax[1,1].set_title('Variance spectra - testing')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#%% test 5 - confidence regions - covariance/correlation spectra

t5 = np.arange(1,301)
sinal5_1_train = t5 +50*np.random.rand(t5.size)
sinal5_2_train = 3*t5 +50*np.random.rand(t5.size)

test5_train = spectra(np.array([sinal5_1_train,sinal5_2_train]).T)

test5_train.calc_Cov_Spectra('sliding',percentiles=[40,50,60])

fig, ax = plt.subplots(2,3,figsize=(12,5))

fig.suptitle('Test 5')

ax[0,0].plot(t5,sinal5_1_train)
ax[0,0].set_prop_cycle(None)
ax[0,0].plot(t5,sinal5_2_train)

test5_train.plot_Cov_Spectra(0,1,ax=ax[0,1],corr_or_cov='cov', percentile=50)
test5_train.plot_Cov_Spectra(0,1,ax=ax[0,1],corr_or_cov='cov',
                             conf_region=[40,60])

ax[1,1].fill_between(np.arange(2,test5_train.spctr_size['sliding'],
                               test5_train.jump_WS_cov['sliding']),
                     test5_train.lim_inf_cov,
                     test5_train.lim_sup_cov,
                     alpha=0.2); 

test5_train.plot_Cov_Spectra(0,1,ax=ax[0,2],corr_or_cov='corr', percentile=50)
test5_train.plot_Cov_Spectra(0,1,ax=ax[0,2],corr_or_cov='corr',
                             conf_region=[40,60])
  
ax[1,2].fill_between(np.arange(2,test5_train.spctr_size['sliding'],
                               test5_train.jump_WS_cov['sliding']),
                     test5_train.lim_inf_cov,
                     test5_train.lim_sup_cov,
                     alpha=0.2);

ax[0,0].set_xlabel('Time')
ax[0,0].set_title('Simulated signals')
ax[0,1].set_title('Covariance spectra')
ax[0,2].set_title('Correlation spectra')

sinal5_1_test = t5 +80*np.random.rand(t5.size)
sinal5_2_test = 3*t5 +80*np.random.rand(t5.size)

test5_test = spectra(np.array([sinal5_1_test,sinal5_2_test]).T)

test5_test.calc_Cov_Spectra('sliding',jump_WS=1,percentiles=[25,50,75])

fig.suptitle('Test 5')

ax[1,0].plot(t5,sinal5_1_test)
ax[1,0].set_prop_cycle(None)
ax[1,0].plot(t5,sinal5_2_test)

test5_test.plot_Cov_Spectra(0,1,ax=ax[1,1],corr_or_cov='cov', percentile=50)

test5_test.plot_Cov_Spectra(0,1,ax=ax[1,2],corr_or_cov='corr',percentile=50)  

ax[1,0].set_xlabel('Time')
ax[1,0].set_title('Simulated signals')
ax[1,1].set_title('Covariance spectra')
ax[1,2].set_title('Correlation spectra')

#ax[0,2].set_ylim(0,1)
#ax[1,2].set_ylim(0,1)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#%% test 6 - confidence regions - latent variance spectra

n=100

t3=np.arange(50)
sinais3 = [j*t3 +np.random.randn(t3.size)*n for j in range(1,11)]

test3 = spectra(np.array(sinais3).T)

test3.calc_Lat_Var_Spectra(percentiles=[25,50,75])

fig, ax = plt.subplots(1,2,figsize=(12,5))

fig.suptitle('Test 3')

[ax[0].plot(t3,sinais3[i]) for i in range(len(sinais3))]

ax[0].set_title('Simulated signals')
ax[0].set_xlabel('t')

test3.plot_Lat_Var_Spectra(ax=ax[1],percentile=50)
test3.plot_Lat_Var_Spectra(ax=ax[1],conf_region=[25,75])

ax[1].set_title('Latent variance spectra')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

#%% test 7 - spectra dynamics

# http://openmv.net/info/flotation-cell
flot = pd.read_csv('http://openmv.net/file/flotation-cell.csv',index_col=0)
flot.index = pd.to_datetime(flot.index,infer_datetime_format=True)

test7 = spectra_dynamics(flot.iloc[:300],WS=50,step=10)
test7.calc_Spectra_Dynamics('cov')
test7.plot_Spectra_Dynamics(stat='corr',i=1,j=3)
test7.plot_Spectra_Dynamics_Lines(stat='corr',i=1,j=3)
test7.plot_Spectra_Dynamics_Video(stat='corr',i=1,j=3)
#test7.anim.save('test7.mp4', metadata={'artist':'Guido'})

