#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 21:23:07 2018

@author: afranio
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from spectra import spectra

#%% signal of example 1

t = np.arange(1,101)
sinal1 = np.sin(t/4)

plt.plot(t,sinal1)
plt.title('Simulated periodic signal - example 1')
plt.xlabel('t')
plt.ylabel('signal');

#%% running example 1

# creating example 1 instance
ex1 = spectra(sinal1)

# calculating example 1 spectrum, with sliding windows
ex1.sliding_Window_Var_Spectra()

# calculating example 1 spectrum, with independent windows
ex1.independent_Window_Var_Spectra()

plt.figure()

# plotting results
ex1.plot_Var_Spectra(plt.gca())
plt.title('Variance spectra - example 1')
plt.xlabel('Window size')
plt.ylabel('$\sigma^2$')

#%% signal of example 2

t = np.arange(1,501)
sinal2 = np.sin(t/4)

plt.figure()
plt.plot(t,sinal2)
plt.title('Simulated periodic signal - example 2')
plt.xlabel('t')
plt.ylabel('signal')

#%% running example 2

ex2 = spectra(sinal2)

ex2.sliding_Window_Var_Spectra()
ex2.independent_Window_Var_Spectra()

plt.figure()

fig2 = ex2.plot_Var_Spectra(plt.gca())
plt.title('Variance spectra - example 2')
plt.xlabel('Window size')
plt.ylabel('$\sigma^2$')

#%% comparing exs 1 and 2

fig, ax = plt.subplots(2,1)

ex1.plot_Var_Spectra(ax[0])
ax[0].axis((20,60,0.45,0.52))
ax[0].set_title('Variance spectra - example 1')

ex2.plot_Var_Spectra(ax[1])
ax[1].axis((20,60,0.45,0.52))
ax[1].set_title('Variance spectra - example 2');

#%% signal of example 3

t = np.arange(-12*np.pi,12*np.pi+0.1,0.5)

sinal3 = 0

for i in range(20):
    sinal3 += np.sin((i+1)*t/4)
    
plt.plot(t,sinal3)

plt.title('Simulated periodic signal - example 3')
plt.xlabel('t')
plt.ylabel('signal');

#%% running example 3

ex3 = spectra(sinal3)

ex3.sliding_Window_Var_Spectra()
ex3.independent_Window_Var_Spectra()

ex3.plot_Var_Spectra(plt.gca())
plt.title('Variance spectra - example 3')
plt.xlabel('Window size')
plt.ylabel('$\sigma^2$')

#%% example 4

t = np.arange(-12*np.pi,12*np.pi+0.1,0.5)

fig, axarr = plt.subplots(1,2,figsize=(12,6))

plt.close()

sinal4 = []
ex4 = []
soma = 0

for i in range(200):
    
    soma += np.sin((i+1)*t/4)
    sinal4.append(soma.copy()) 
    ex4.append(spectra(soma))
    ex4[i].sliding_Window_Var_Spectra()
    ex4[i].independent_Window_Var_Spectra()

def animate(i):

    axarr[0].clear()
    axarr[0].set_title('Simulated periodic signal')
    axarr[0].set_xlabel('t')
    axarr[0].set_ylabel('signal');
    axarr[1].clear()
    axarr[1].set_title('Variance spectra')
    axarr[1].set_xlabel('Window size')
    axarr[1].set_ylabel('$\sigma^2$');
    axarr[0].plot(t,sinal4[i])
    ex4[i].plot_Var_Spectra(axarr[1])
    fig.suptitle('Example 4')
    
    return (axarr[0],axarr[1])

anim4 = animation.FuncAnimation(fig, animate, frames=200, interval=300, blit=False)
#anim4.save('anim4.mp4')

#%% example 5

sinal5_1 = np.arange(1,101)
sinal5_2 = 3*sinal5_1

ex5 = spectra(sinal5_1,sinal5_2)

ex5.sliding_Window_Cov_Spectra()
ex5.independent_Window_Cov_Spectra()

plt.figure()

ex5.plot_Cov_Spectra(plt.gca())
plt.title('Covariance spectra - example 5')
plt.xlabel('Window size')
plt.ylabel('$\sigma_{ij}^2$')

plt.figure()

ex5.plot_Corr_Spectra(plt.gca())
plt.title('Correlation spectra - example 5')
plt.xlabel('Window size')
plt.ylabel('$r_{ij}$')

#%% example 6

sinal6_1 = np.arange(1,501)
sinal6_2 = sinal6_1 + np.random.rand(sinal6_1.size)*sinal6_1

ex6 = spectra(sinal6_1,sinal6_2)

ex6.sliding_Window_Cov_Spectra()
ex6.independent_Window_Cov_Spectra()

plt.figure()

ex6.plot_Cov_Spectra(plt.gca())
plt.title('Covariance spectra - example 6')
plt.xlabel('Window size')
plt.ylabel('$\sigma_{ij}^2$')

plt.figure()

ex6.plot_Corr_Spectra(plt.gca())
plt.title('Correlation spectra - example 6')
plt.xlabel('Window size')
plt.ylabel('$r_{ij}$')

#%% example 7

t = np.arange(1,101)
sinal7_1 = np.sin(t/4)

sinal7_2 = np.cos(t/4)

ex7 = spectra(sinal7_1,sinal7_2)

ex7.sliding_Window_Cov_Spectra()
ex7.independent_Window_Cov_Spectra()

ex7.plot_Cov_Spectra(plt.gca())
plt.title('Covariance spectra - example 7')
plt.xlabel('Window size')
plt.ylabel('$\sigma_{ij}^2$')

plt.figure()

ex7.plot_Corr_Spectra(plt.gca())
plt.title('Correlation spectra - example 7')
plt.xlabel('Window size')
plt.ylabel('$r_{ij}$')

#%% example 8

t = np.arange(1,51)

sinal8_1 = np.sin(t/4)
sinal8_2 = []
ex8 = []

for i in range(128):
    
    sinal8_2.append(np.cos(t/4+i*(np.pi/64)))
    ex8.append(spectra(sinal8_1, sinal8_2[i]))
    ex8[i].sliding_Window_Cov_Spectra()
    ex8[i].independent_Window_Cov_Spectra()

fig, axarr = plt.subplots(1,3,figsize=(13,6))
plt.close()

def animate(i):
    
    axarr[0].clear()
    axarr[0].set_title('Simulated periodic signal')
    axarr[0].set_xlabel('t')
    axarr[0].set_ylabel('signal');
    axarr[1].clear()
    axarr[1].set_title('Covariance spectra')
    axarr[1].set_xlabel('Window size')
    axarr[1].set_ylabel('$\sigma_{ij}^2$')
    axarr[2].clear()
    axarr[2].set_title('Correlation spectra')
    axarr[2].set_xlabel('Window size')
    axarr[2].set_ylabel('$r_{ij}$');
    axarr[0].plot(t,sinal8_1)
    axarr[0].plot(t,sinal8_2[i])
    ex8[i].plot_Cov_Spectra(axarr[1])
    ex8[i].plot_Corr_Spectra(axarr[2])
    axarr[0].set_xlim(0,50)
    axarr[0].set_ylim(-1,1)
    axarr[1].set_xlim(0,50)
    axarr[1].set_ylim(-0.75,0.75)
    axarr[2].set_xlim(0,50)
    axarr[2].set_ylim(-1,1)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.suptitle('Example 8')

    return axarr

anim8 = animation.FuncAnimation(fig, animate,frames=128, interval=200, blit=False)
#anim8.save('anim8.mp4')
