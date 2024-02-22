# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:06:41 2022

@author: admin
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from glob import glob
import scanpy as sc
import sys
from dca.api import dca
import os
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_path)
from utils import Mkdir

def Cpm_added(count, features):
    cpm = count
    addrows = np.setdiff1d(features, cpm.index)
    addmat = pd.DataFrame(0.01, index=addrows, columns=cpm.columns)
    cpm_added = pd.concat([cpm, addmat])
    
    for i in cpm_added.columns:
    	cpm_added.loc[:,i] = cpm_added.loc[:,i]/(sum(cpm_added.loc[:,i]))*(1e6)
    cpm_added = cpm_added.loc[features,:]
    return cpm_added

scaler = StandardScaler()

def Zcol(cpm_added): 
    cpm_log = np.log2(cpm_added+1)
    cpm_zcol = cpm_log
    for i in cpm_zcol.columns:
    	cpm_zcol[[i]] = scaler.fit_transform(cpm_zcol[[i]])
    return cpm_zcol


def GetFiles(filepath, fileclass):
    if fileclass in ['h5', 'txt', 'csv', 'h5ad']:
        files = glob(filepath+r'*.'+fileclass)
        files = [c.replace('\\','/') for c in files]
        filenames = [c.split('/')[-1] for c in files]
        filenames = [c.replace('.'+fileclass,'') for c in filenames]
    #elif fileclass == '10x':
    #    files = glob(filepath+r'*.matrix.mtx')
    #    files = [c.replace('\\','/') for c in files]
    #    filenames = [c.split('/')[-1] for c in files]
    #    filenames = [c.replace('.matrix.mtx') for c in filenames]        
    else:
        raise Exception('File should be one of h5, txt(tab separated), csv, h5ad')
    print(str(len(filenames))+' files detected: ')
    for i, filename in enumerate(filenames):
        print(str(i), '\t', filename)        
    return files, filenames

def MakeAdata(file, filename, fileclass, add_prefix = False, genenameCol = None):
    print(filename+', making adata...') 
    if fileclass == 'h5':
        adata = sc.read_10x_h5(file)
    #elif fileclass in ['10x', '10x_v3']:
    #    filedir = '/'.join(file.split('/')[:-1])
    #    adata = sc.read_10x_mtx(
    #        filedir,  # the directory with thes `.mtx` file
    #        var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    #        cache=True, 
    #        prefix=filename+'.') 
    elif fileclass == 'csv':
        data = pd.read_csv(file, index_col = 0, header=0).T
        adata = sc.AnnData(data)
    elif fileclass == 'txt':
        data = pd.read_csv(file, sep = '\t', index_col = 0, header=0).T
        adata = sc.AnnData(data)
    elif fileclass == 'h5ad':
        adata = sc.read_h5ad(file)    
    if add_prefix:
        adata.obs_names=filename+'_'+adata.obs_names
    if genenameCol is not None:
        adata.var_names = [str(genename) for genename in adata.var.loc[:,genenameCol]]
    adata.var_names_make_unique()
    return adata

def DCA_mat(adata_ae, threads = 1): 
    dca(adata_ae, threads=threads)
    subcount = pd.DataFrame(adata_ae.X)
    subcount.index = adata_ae.obs_names
    subcount.columns = adata_ae.var_names
    subdata_count = subcount.T
    return subdata_count
   

def ScaleData(adata, denoising = True, threads = 1, savetmp = False):
    sene = pd.read_csv(str(base_path)+'/resource/seneset.txt', sep = '\t', index_col = None, header=None)
    sene.columns = ['Gene']
    #sene.Gene = 'GRCh38premrna_'+sene.Gene
    print('Scaling data...')
    seneGenes = list(set(adata.var_names) & set(sene.Gene))
    assert len(seneGenes)>10, 'Features of count data should be hgnc symbol: BIRC5, BRCA1, etc'
    adata_ae = adata[:,seneGenes].copy()
    sc.pp.filter_genes(adata_ae, min_counts=1)
    try:
        adata_ae.X = adata_ae.X.toarray()
    except:
        print('No need to convert to array')
    
    subcount_raw = pd.DataFrame(adata_ae.X)
    subcount_raw.index = adata_ae.obs_names
    subcount_raw.columns = adata_ae.var_names
        
    if denoising:    
        subdata_count = DCA_mat(adata_ae, threads = threads)
    else: 
        subdata_count = subcount_raw.T

    cpm_added = Cpm_added(subdata_count, sene.Gene)
    cpm_zcol = Zcol(cpm_added)
    if savetmp:
        tmps = [subcount_raw.T, subdata_count]
    else:
        tmps = None
    return cpm_zcol, tmps

