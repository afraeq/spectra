#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class spectra (object):

    ###########################################################################

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
        self.l_var_spctr = {}
        
        self.jump_WS_var   = {}
        self.jump_WS_cov   = {}
        self.jump_WS_l_var = {}
        
        kinds = ['sliding','independent']
        
        for kind in kinds:       
            self.var_spctr   [kind] = {}
            self.cov_spctr   [kind] = {}
            self.corr_spctr  [kind] = {}
            self.l_var_spctr [kind] = {}

        self.spctr_size = {}            
        self.spctr_size['sliding']     = self.data.shape[0]+1
        self.spctr_size['independent'] = int(self.data.shape[0]/2+1)
        
        # https://stackoverflow.com/questions/39232790
        self.idx = lambda WS,d: \
        (d*np.arange((self.data.shape[0]-WS+1)/d)[:,None].astype(int) + \
        np.arange(WS))

        self.d = {}
        self.d['sliding']     = lambda WS: 1
        self.d['independent'] = lambda WS: WS
        
    ###########################################################################
                
    def calc_Var_Spectra (self, kind='sliding', jump_WS=1, 
                          percentiles=[]):
        
        self.jump_WS_var[kind] = jump_WS
                
        spctr_size = self.spctr_size[kind]
                        
        var_spctr_mean = np.zeros((spctr_size,self.data.shape[1]))

        var_spctr_percentile = {}
        
        for i in percentiles:
            var_spctr_percentile[i] = np.zeros((spctr_size,self.data.shape[1]))
        
        for WS in range(2,spctr_size,jump_WS):
            
            win = self.data[self.idx(WS,self.d[kind](WS))]
            var = np.var(win,ddof=1,axis=1)
            var_spctr_mean[WS,:]   = np.mean(var,axis=0)
            
            for i in percentiles:
                var_spctr_percentile[i][WS,:] = np.percentile(var, i, axis=0)
                                
        self.var_spctr[kind]['mean']   = var_spctr_mean                                             
        for i in percentiles:
            self.var_spctr[kind][i] = var_spctr_percentile[i]
            
    ###########################################################################
    
    def calc_Cov_Spectra (self, kind='sliding', jump_WS=1,
                          percentiles=[]):
        
        self.jump_WS_cov[kind] = jump_WS
        
        spctr_size = self.spctr_size[kind]
        
        cov_spctr_mean  = np.zeros((spctr_size,self.data.shape[1],
                                    self.data.shape[1]))
        corr_spctr_mean = np.zeros((spctr_size,self.data.shape[1],
                                    self.data.shape[1]))

        cov_spctr_percentile  =  {}
        corr_spctr_percentile =  {}
        
        for i in percentiles:
            cov_spctr_percentile [i] = np.zeros((spctr_size,self.data.shape[1],
                                                self.data.shape[1]))
            corr_spctr_percentile[i] = np.zeros((spctr_size,self.data.shape[1],
                                                self.data.shape[1]))

        # https://stackoverflow.com/questions/26089893
        # https://stackoverflow.com/questions/40394775
        
        for WS in range(2,spctr_size,jump_WS):  
            
            win = self.data[self.idx(WS,self.d[kind](WS))]
            m1 = win - win.sum(axis=1,keepdims=True)/win.shape[1]
            Sxx = np.einsum('ijk,ijl->ikl',m1,m1)/(win.shape[1] - 1)
            Sxx_mean = np.einsum('ijk->jk',Sxx)/win.shape[0]
            cov_spctr_mean[WS,:,:] = Sxx_mean
            Dinv = np.linalg.inv(np.diag(np.sqrt(np.diag(Sxx_mean))))
            corr_spctr_mean[WS,:,:] = Dinv@Sxx_mean@Dinv
            
            for i in percentiles:
                Sxx_perc = np.percentile(Sxx, i, axis=0)
                cov_spctr_percentile[i][WS,:]  = Sxx_perc
                Dinv = np.linalg.inv(np.diag(np.sqrt(np.diag(Sxx_perc))))
                corr_spctr_percentile[i][WS,:] = Dinv@Sxx_perc@Dinv

        self.cov_spctr[kind]['mean']  = cov_spctr_mean
        self.corr_spctr[kind]['mean'] = corr_spctr_mean
        
        for i in percentiles:
            self.cov_spctr[kind][i]  = cov_spctr_percentile[i]
            self.corr_spctr[kind][i] = corr_spctr_percentile[i]

    ###########################################################################
    
    def calc_Lat_Var_Spectra (self, kind='sliding',jump_WS=1,
                              percentiles=[]):
        
        self.jump_WS_l_var[kind] = jump_WS
        
        from sklearn.preprocessing import scale
        data = scale(self.data)

        spctr_size = self.spctr_size[kind]
        
        l_var_spctr_mean = np.zeros((spctr_size,data.shape[1]))
        
        l_var_spctr_percentile  = {}
        
        for i in percentiles:
            l_var_spctr_percentile [i] = np.zeros((spctr_size,data.shape[1]))
                
        for WS in range(data.shape[1],spctr_size,jump_WS):  
            
            win = data[self.idx(WS,self.d[kind](WS))]
            m1 = win - win.sum(axis=1,keepdims=True)/win.shape[1]
            Sxx = np.einsum('ijk,ijl->ikl',m1,m1)/(win.shape[1] - 1)
            Sxx_mean = np.einsum('ijk->jk',Sxx)/win.shape[0]
            _, L, _ = np.linalg.svd(Sxx_mean)
            l_var_spctr_mean[WS,:] = L
            
            for i in percentiles:
                Sxx_perc = np.percentile(Sxx, i, axis=0)
                _, L, _ = np.linalg.svd(Sxx_perc)
                l_var_spctr_percentile[i][WS,:] = L
                                            
        self.l_var_spctr[kind]['mean'] = l_var_spctr_mean
        
        for i in percentiles:
            self.l_var_spctr[kind][i] = l_var_spctr_percentile[i]
        
    ###########################################################################

    def plot_Var_Spectra(self,i=None,ax=None,
                         percentile='mean',
                         conf_region=False):
        
        if i==None:
            i= np.arange(self.data.shape[1])
        if ax == None:
            ax = plt.gca() 
        if self.var_spctr['sliding']:
            ax.set_prop_cycle(None)
            ax.plot(np.arange(2,self.spctr_size['sliding'],
                              self.jump_WS_var['sliding']),
                    self.var_spctr['sliding']
                                  [percentile]
                                  [2::self.jump_WS_var['sliding'],i])
        if self.var_spctr['independent']:
            ax.set_prop_cycle(None)
            ax.plot(np.arange(2,self.spctr_size['independent'],
                              self.jump_WS_var['independent']),
                    self.var_spctr['independent']
                                  [percentile]
                                  [2::self.jump_WS_var['independent'],i],'.')
        ax.set_xlabel('Window size')
        ax.set_ylabel('$\sigma^2$')
        ax.set_xticks(list(ax.get_xticks()) + [2])
        ax.margins(0);
        
        if conf_region:
            
            ax.set_prop_cycle(None)
            
            for i in range(self.data.shape[1]):

                ax.fill_between(np.arange(2,self.spctr_size['sliding'],
                                          self.jump_WS_var['sliding']),
                                self.var_spctr['sliding']
                                              [conf_region[0]]
                                              [2::self.jump_WS_var['sliding'],
                                               i].squeeze(),
                                self.var_spctr['sliding']
                                              [conf_region[1]]
                                              [2::self.jump_WS_var['sliding'],
                                               i].squeeze(), 
                                alpha=0.2);
            
    ###########################################################################
    
    def plot_Cov_Spectra(self,i,j,ax=None,
                         corr_or_cov='cov',
                         percentile='mean',
                         conf_region=False):

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
            ax.plot(np.arange(2,self.spctr_size['sliding'],
                              self.jump_WS_cov['sliding']),
                    spctr['sliding'][percentile]
                         [2::self.jump_WS_cov['sliding'],i,j])
        if spctr['independent']:
            ax.set_prop_cycle(None)
            ax.plot(np.arange(2,self.spctr_size['independent'],
                              self.jump_WS_cov['independent']),
                    spctr['independent'][percentile]
                         [2::self.jump_WS_cov['independent'],i,j],'.')
        ax.set_xticks(list(ax.get_xticks()) + [2])
        ax.set_xlabel('Window size')
        ax.set_ylabel(ylabel)
        ax.margins(0);
        
        if conf_region:
            ax.set_prop_cycle(None)
            ax.fill_between(np.arange(2,self.spctr_size['sliding'],
                                          self.jump_WS_cov['sliding']),
                                spctr['sliding']
                                     [conf_region[0]]
                                     [2::self.jump_WS_cov['sliding'],i,j].\
                                     squeeze(),
                                spctr['sliding']
                                     [conf_region[1]]
                                     [2::self.jump_WS_cov['sliding'],i,j].\
                                     squeeze(),
                                alpha=0.2);
        
    ###########################################################################
    
    def plot_Lat_Var_Spectra(self,ax=None,percentile='mean',
                             conf_region=False):
        
        if ax == None:
            ax = plt.gca() 
        
        if self.l_var_spctr['independent']:
            ax.set_prop_cycle(None)
            ax.plot(np.arange(self.data.shape[1],
                              self.spctr_size['independent'],
                              self.jump_WS_l_var['independent']),
                    self.l_var_spctr['independent']
                                    [percentile]
                                    [self.data.shape[1]::
                                     self.jump_WS_l_var['independent']],'.')
        if self.l_var_spctr['sliding']:
            ax.set_prop_cycle(None)
            ax.plot(np.arange(self.data.shape[1],
                              self.spctr_size['sliding'],
                              self.jump_WS_l_var['sliding']),
                    self.l_var_spctr['sliding']
                                    [percentile]
                                    [self.data.shape[1]::
                                     self.jump_WS_l_var['sliding']])            
        ax.set_xticks(list(ax.get_xticks()) + [self.data.shape[1]])
        ax.set_xlabel('Window size')
        ax.set_ylabel('$\lambda_i$')
        ax.margins(0);
        
        if conf_region:
            ax.set_prop_cycle(None) 
            for i in range(self.data.shape[1]):    
                ax.fill_between(np.arange(2,self.spctr_size['sliding'],
                                          self.jump_WS_l_var['sliding']),
                                self.l_var_spctr['sliding']
                                                [conf_region[0]]
                                             [2::self.jump_WS_l_var['sliding'],
                                              i].squeeze(),
                                self.l_var_spctr['sliding']
                                                [conf_region[1]]
                                             [2::self.jump_WS_l_var['sliding'],
                                              i].squeeze(),
                                alpha=0.2);                            