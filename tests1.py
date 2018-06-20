#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 21:23:07 2018

@author: afranio
"""

import numpy as np
import matplotlib.pyplot as plt

from spectra import spectra

#%% signal of test 1

t = np.arange(1,101)
sinal1 = np.sin(t/4)

plt.plot(t,sinal1)
plt.title('Simulated periodic signal - test 1')
plt.xlabel('t')
plt.ylabel('signal');

#%% running test 1

# creating test 1 instance
test1 = spectra(sinal1)

# calculating test 1 spectrum, with sliding windows
test1.sliding_Window_Var_Spectra()

# calculating test 1 spectrum, with independent windows
test1.independent_Window_Var_Spectra()

plt.figure()

# plotting results
test1.plot_Var_Spectra(plt.gca())
plt.title('Variance spectra - test 1')

#%% signal of test 2

t = np.arange(1,501)
sinal2 = np.sin(t/4)

plt.figure()
plt.plot(t,sinal2)
plt.title('Simulated periodic signal - test 2')
plt.xlabel('t')
plt.ylabel('signal');

#%% running test 2

test2 = spectra(sinal2)

test2.sliding_Window_Var_Spectra()
test2.independent_Window_Var_Spectra()

plt.figure()

fig2 = test2.plot_Var_Spectra(plt.gca())
plt.title('Variance spectra - test 2');

#%% comparing test 1 and 2

fig, ax = plt.subplots(2,1)

test1.plot_Var_Spectra(ax[0])
ax[0].axis((20,60,0.45,0.52))
ax[0].set_title('Variance spectra - test 1')

test2.plot_Var_Spectra(ax[1])
ax[1].axis((20,60,0.45,0.52))
ax[1].set_title('Variance spectra - test 2');

#%% test 3

sinal1 = np.arange(1,101)
sinal2 = 3*sinal1

test3 = spectra(sinal1,sinal2)

test3.sliding_Window_Cov_Spectra()
test3.independent_Window_Cov_Spectra()

plt.figure()

test3.plot_Cov_Spectra(plt.gca())
plt.title('Covariance spectra - test 3')

plt.figure()

test3.plot_Corr_Spectra(plt.gca())
plt.title('Correlation spectra - test 3');

#%% test4

sinal1 = np.arange(1,501)
sinal2 = sinal1 + np.random.rand(sinal1.size)*sinal1

test4 = spectra(sinal1,sinal2)

test4.sliding_Window_Cov_Spectra()
test4.independent_Window_Cov_Spectra()

plt.figure()

test4.plot_Cov_Spectra(plt.gca())
plt.title('Covariance spectra - test 4')

plt.figure()

test4.plot_Corr_Spectra(plt.gca())
plt.title('Correlation spectra - test 4');

#%% test5

t = np.arange(1,101)
sinal1 = np.sin(t/4)

sinal2 = np.cos(t/4)

test5 = spectra(sinal1,sinal2)

test5.sliding_Window_Cov_Spectra()
test5.independent_Window_Cov_Spectra()

plt.figure()

test5.plot_Cov_Spectra(plt.gca())
plt.title('Covariance spectra - test 5')

plt.figure()

test5.plot_Corr_Spectra(plt.gca())
plt.title('Correlation spectra - test 5');
