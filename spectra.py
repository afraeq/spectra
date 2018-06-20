#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class spectra (object):
    
    #########################
    
    def __init__ (self,data,data2=[]):
        
        # dados a serem analisados
        self.data = np.squeeze(np.asarray(data, dtype=np.float64))
        self.data2 = np.squeeze(np.asarray(data2, dtype=np.float64))
        
        # número de pontos
        self.ND = self.data.size
        self.ND2 = self.data2.size

    #########################
        
    def sliding_Window_Var_Spectra (self):
        
        # variável que armazenará o espectro
        self.spctr_var_sliding = np.zeros(self.ND)
                
        # para cada tamanho de janela...
        for TJ in range (2,self.ND):
            
            # ... calcule o número de janelas ...
            NJ = self.ND - TJ +1
            
            # ... inicialize a soma ...
            soma = 0
            
            # ... e calcule o somatorio das variâncias de cada janela
            for i in range(0,NJ):
                soma += np.var(self.data[i:i+TJ])
                
            self.spctr_var_sliding[TJ] = soma/NJ
                                            
    #########################

    def independent_Window_Var_Spectra (self):
        
        # variável que armazenará o espectro
        self.spctr_var_ind = np.zeros(int(self.ND/2))
        
        # para cada tamanho de janela...
        for TJ in range (2,int(self.ND/2)):

            # ... calcule o número de janelas ...
            NJ = int(self.ND/TJ)
            
            # ... inicialize a soma ...
            soma = 0
            
            # ... e calcule o somatorio das variâncias de cada janela
            for i in range(NJ):
                soma += np.var(self.data[i*TJ:(i+1)*TJ])
                
            self.spctr_var_ind[TJ] = soma/NJ
            
    #########################
            
    def sliding_Window_Cov_Spectra (self):
        
        # variável que armazenará o espectro
        self.spctr_cov_sliding = np.zeros(self.ND)
        self.spctr_corr_sliding = np.zeros(self.ND)
                
        # para cada tamanho de janela...
        for TJ in range (2,self.ND):
            
            # ... calcule o número de janelas ...
            NJ = self.ND - TJ +1
            
            # ... inicialize as somas ...
            soma1 = 0
            soma2 = 0
            
            # ... e calcule os somatorios das covariâncias e das correlações de cada janela
            for i in range(0,NJ):
                soma1 += np.cov(self.data[i:i+TJ],self.data2[i:i+TJ])[0,1]
                soma2 += np.corrcoef(self.data[i:i+TJ],self.data2[i:i+TJ])[0,1]

            self.spctr_cov_sliding[TJ] = soma1/NJ
            self.spctr_corr_sliding[TJ] = soma2/NJ
                                            
    #########################
    
    def independent_Window_Cov_Spectra (self):
        
        # variável que armazenará o espectro
        self.spctr_cov_ind = np.zeros(int(self.ND/2))
        self.spctr_corr_ind = np.zeros(int(self.ND/2))
        
        # para cada tamanho de janela...
        for TJ in range (2,int(self.ND/2)):

            # ... calcule o número de janelas ...
            NJ = int(self.ND/TJ)
            
            # ... inicialize a soma ...
            soma1 = 0
            soma2 = 0
            
            # ... e calcule os somatorios das covariâncias e das correlações de cada janela
            for i in range(NJ):
                soma1 += np.cov(self.data[i*TJ:(i+1)*TJ],self.data2[i*TJ:(i+1)*TJ])[0,1]
                soma2 += np.corrcoef(self.data[i*TJ:(i+1)*TJ],self.data2[i*TJ:(i+1)*TJ])[0,1]
            
            self.spctr_cov_ind[TJ] = soma1/NJ
            self.spctr_corr_ind[TJ] = soma2/NJ
                
    #########################
    
    def plot_Var_Spectra (self, ax):
                
        if hasattr (self, 'spctr_var_sliding') and hasattr (self, 'spctr_var_ind'):
        
            ax.plot(np.arange(len(self.spctr_var_sliding)),self.spctr_var_sliding,label='Janela deslizante')
            ax.plot(np.arange(len(self.spctr_var_ind)),self.spctr_var_ind,'.',label='Janelas independentes')
            
        elif hasattr (self,'spctr_var_sliding') and not hasattr (self, 'spctr_var_ind'):
              
            ax.plot(np.arange(len(self.spctr_var_sliding)),self.spctr_var_sliding)

        elif hasattr (self,'spctr_var_ind') and not hasattr (self, 'spctr_var_sliding'):
              
            ax.plot(np.arange(len(self.spctr_var_ind)),self.spctr_var_ind,'.')
                        
    #########################            
            
    def plot_Cov_Spectra (self, ax):
                
        if hasattr (self, 'spctr_cov_sliding') and hasattr (self, 'spctr_cov_ind'):
        
            ax.plot(np.arange(len(self.spctr_cov_sliding)),self.spctr_cov_sliding,label='Janela deslizante')
            ax.plot(np.arange(len(self.spctr_cov_ind)),self.spctr_cov_ind,'.',label='Janelas independentes')
            
        elif hasattr (self,'spctr_cov_sliding') and not hasattr (self, 'spctr_cov_ind'):
              
            ax.plot(np.arange(len(self.spctr_cov_sliding)),self.spctr_cov_sliding,'.')

        elif hasattr (self,'spctr_cov_ind') and not hasattr (self, 'spctr_cov_sliding'):
              
            ax.plot(np.arange(len(self.spctr_cov_ind)),self.spctr_cov_ind,'.')

    #########################            
            
    def plot_Corr_Spectra (self, ax):
                
        if hasattr (self, 'spctr_corr_sliding') and hasattr (self, 'spctr_corr_ind'):
        
            ax.plot(np.arange(len(self.spctr_corr_sliding)),self.spctr_corr_sliding,label='Janela deslizante')
            ax.plot(np.arange(len(self.spctr_corr_ind)),self.spctr_corr_ind,'.',label='Janelas independentes')
            
        elif hasattr (self,'spctr_corr_sliding') and not hasattr (self, 'spctr_corr_ind'):
              
            ax.plot(np.arange(len(self.spctr_corr_sliding)),self.spctr_corr_sliding,'.')

        elif hasattr (self,'spctr_corr_ind') and not hasattr (self, 'spctr_corr_sliding'):
              
            ax.plot(np.arange(len(self.spctr_corr_ind)),self.spctr_corr_ind,'.')