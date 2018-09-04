#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class spectra (object):

    #########################
    
    def __init__ (self,data):
               
        data = np.asarray(data)
            
        if data.ndim==1:
            self.data = data[:,np.newaxis]
        elif data.ndim==2:
                self.data = data
        else:
            raise ValueError('Entry must have 1 or 2 dimensions!')
        
        self.var_spctr   = {}
        self.cov_spctr   = {}
        self.corr_spctr  = {}
        self.m_var_spctr = {}
        
        kinds = ['sliding','independent']
        
        for kind in kinds:       
            self.var_spctr   [kind] = {}
            self.cov_spctr   [kind] = {}
            self.corr_spctr  [kind] = {}
            self.m_var_spctr [kind] = {}

        self.spctr_size = {}            
        self.spctr_size['sliding']     = self.data.shape[0]+1
        self.spctr_size['independent'] = int(self.data.shape[0]/2+1)
        
        # https://stackoverflow.com/questions/39232790
        self.idx = lambda WS,d: (d*np.arange((self.data.shape[0]-WS+1)/d)[:,None].astype(int) + np.arange(WS))

        self.d = {}
        self.d['sliding']     = lambda WS: 1
        self.d['independent'] = lambda WS: WS
        
    #########################
                
    def calc_Var_Spectra (self,kind):
                
        spctr_size = self.spctr_size[kind]
                        
        var_spctr_mean = np.zeros((spctr_size,self.data.shape[1]))
        var_spctr_median = np.zeros((spctr_size,self.data.shape[1]))
        
        for WS in range(2,spctr_size):
            win = self.data[self.idx(WS,self.d[kind](WS))]
            var = np.var(win,ddof=1,axis=1)
            var_spctr_mean[WS,:]   = np.mean(var,axis=0)
            var_spctr_median[WS,:] = np.median(var,axis=0)
                        
        self.var_spctr[kind]['mean']   = var_spctr_mean                                             
        self.var_spctr[kind]['median'] = var_spctr_median

    ################################################ 
    
    def calc_Cov_Spectra (self, kind='sliding'):
        
        spctr_size = self.spctr_size[kind]
        
        cov_spctr_mean  = np.zeros((spctr_size,self.data.shape[1],self.data.shape[1]))
        corr_spctr_mean = np.zeros((spctr_size,self.data.shape[1],self.data.shape[1]))
                
        # https://stackoverflow.com/questions/26089893
        # https://stackoverflow.com/questions/40394775
        
        for WS in range(2,spctr_size):  
            win = self.data[self.idx(WS,self.d[kind](WS))]
            m1 = win - win.sum(axis=1,keepdims=True)/win.shape[1]
            Sxx = np.einsum('ijk,ijl->ikl',m1,m1)/(win.shape[1] - 1)
            Sxx_mean = np.einsum('ijk->jk',Sxx)/win.shape[0]
            cov_spctr_mean[WS,:,:] = Sxx_mean
            Dinv = np.linalg.inv(np.diag(np.sqrt(np.diag(Sxx_mean))))
            corr_spctr_mean[WS,:,:] = Dinv@Sxx_mean@Dinv
            
        self.cov_spctr[kind]['mean']  = cov_spctr_mean
        self.corr_spctr[kind]['mean'] = corr_spctr_mean

    #########################   
    
    def calc_Multi_Var_Spectra (self, kind='sliding'):
        
        from sklearn.preprocessing import scale
        data = scale(self.data)

        spctr_size = self.spctr_size[kind]
        
        m_var_spctr_mean = np.zeros((spctr_size,data.shape[1]))
                
        for WS in range(data.shape[1],spctr_size):  
            win = data[self.idx(WS,self.d[kind](WS))]
            m1 = win - win.sum(axis=1,keepdims=True)/win.shape[1]
            Sxx = np.einsum('ijk,ijl->ikl',m1,m1)/(win.shape[1] - 1)
            Sxx_mean = np.einsum('ijk->jk',Sxx)/win.shape[0]
            _, L, _ = np.linalg.svd(Sxx_mean)
            m_var_spctr_mean[WS,:] = L
            
        self.m_var_spctr[kind]['mean'] = m_var_spctr_mean
        
    #########################   

    def plot_Var_Spectra(self,i=None,ax=None,mean_or_median='mean'):
        
        if i==None:
            i= np.arange(self.data.shape[1])
        if ax == None:
            ax = plt.gca() 
        if self.var_spctr['sliding']:
            ax.set_prop_cycle(None)
            pd.DataFrame(self.var_spctr['sliding'][mean_or_median][:,i]).plot(ax=ax)
        if self.var_spctr['independent']:
            ax.set_prop_cycle(None)
            pd.DataFrame(self.var_spctr['independent'][mean_or_median][:,i]).plot(ax=ax,
                                                                                  linestyle='',
                                                                                  marker='.')
        ax.set_xlabel('Window size')
        ax.set_ylabel('$\sigma^2$')
        ax.legend_.remove()
        ax.margins(0);
        
    ######################### 
    
    def plot_Cov_Spectra(self,i,j,ax=None,mean_or_median='mean',corr_or_cov='corr'):

        if ax == None:
            ax = plt.gca()        
            
        if corr_or_cov == 'corr':
            spctr = self.corr_spctr
            ylabel='$r_{ij}$'
        elif corr_or_cov == 'cov':
            spctr = self.cov_spctr
            ylabel='$\sigma_{ij}^2$'    
            
        if spctr['sliding']:
            ax.set_prop_cycle(None)
            pd.DataFrame(spctr['sliding'][mean_or_median][:,i,j]).plot(ax=ax);
        if spctr['independent']:
            ax.set_prop_cycle(None)
            pd.DataFrame(spctr['independent'][mean_or_median][:,i,j]).plot(ax=ax,
                                                                           linestyle='',
                                                                           marker='.')
        ax.set_xlabel('Window size')
        ax.set_ylabel(ylabel)
        ax.legend_.remove()
        ax.margins(0);
        
    ######################### 
    
    def plot_Multi_Var_Spectra(self,ax=None,mean_or_median='mean'):
        
        if ax == None:
            ax = plt.gca() 
        
        if self.m_var_spctr['independent']:
            ax.set_prop_cycle(None)
            pd.DataFrame(self.m_var_spctr['independent'][mean_or_median]).plot(ax=ax,linestyle='',marker='.')
            ax.set_xlim([self.data.shape[1],self.m_var_spctr['independent'][mean_or_median].shape[0]])            
        if self.m_var_spctr['sliding']:
            ax.set_prop_cycle(None)
            pd.DataFrame(self.m_var_spctr['sliding'][mean_or_median]).plot(ax=ax)
            ax.set_xlim([self.data.shape[1],self.m_var_spctr['sliding'][mean_or_median].shape[0]])
        ax.set_xlabel('Window size')
        ax.set_ylabel('$\lambda_i$')
        ax.legend_.remove()
        ax.margins(0);
