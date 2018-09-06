#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 21:23:07 2018

@author: afranio
"""

import numpy as np
import matplotlib.pyplot as plt

from spectra import spectra

#%% test 1

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


#%% test 2

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

#%% test 3

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
