#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 10:11:09 2023

@author: apramanik
"""


import pickle
import numpy as np
import yaml
import argparse
import os
import h5py as h5
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat







def preprocess_training_data(path, num_sl, start_sl, acc):
        
    
    filename = path + 'dataset.hdf5'


    with h5.File(filename) as f:
        org, csm, mask = f['trnOrg'][:], f['trnCsm'][:], f['trnMask'][:]
        
    
    org = org[start_sl: start_sl + num_sl]
    csm = csm[start_sl: start_sl + num_sl]
    mask = mask[start_sl: start_sl + num_sl]
    
    
    org = np.expand_dims(org, axis=1)
    
    mask = mask.astype(np.complex64)
    
    for i in range(mask.shape[0]):
        mask[i] = np.fft.fftshift(mask[i])
    
    
    img = org * csm
    
    ksp = np.zeros_like(img)
    
    us_ksp = np.zeros_like(img)
    
    ncoils = ksp.shape[1]   
    
    for i in range(img.shape[0]):
        
        maxval = np.abs(img[i]).max()
        
        img[i] = img[i]/maxval
        
        ksp[i] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img[i],axes=(-1,-2)), norm='ortho', axes=(-1,-2)), axes=(-1,-2))
    
        us_ksp[i] = ksp[i] * mask[i]
        
        temp = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ksp[i],axes=(-1,-2)), norm='ortho', axes=(-1,-2)), axes=(-1,-2))
    
        org[i] = np.sum(temp*np.conj(csm[i]), axis=0, keepdims=True)
    
    
    
    mask = np.tile(np.expand_dims(mask, axis=1), (1,ncoils,1,1))
    
    org = torch.tensor(org)
    us_ksp = torch.tensor(us_ksp)
    mask = torch.tensor(mask)
    csm = torch.tensor(csm)
    
    
    del temp, ksp, img
    
    
    return org, us_ksp, csm, mask



def preprocess_testing_data(path, num_sl, start_sl, acc):
    
    
    
    filename = path + 'demoImage.hdf5'


    with h5.File(filename) as f:
        org, csm, mask = f['tstOrg'][:], f['tstCsm'][:], f['tstMask'][:]
        
    
    org = org[start_sl: start_sl + num_sl]
    csm = csm[start_sl: start_sl + num_sl]
    mask = mask[start_sl: start_sl + num_sl]
    
    
    org = np.expand_dims(org, axis=1)
    
    mask = mask.astype(np.complex64)
    
    for i in range(mask.shape[0]):
        mask[i] = np.fft.fftshift(mask[i])
    
    img = org * csm
    
    ksp = np.zeros_like(img)
    
    us_ksp = np.zeros_like(img)
    
    ncoils = ksp.shape[1]   
    
    for i in range(img.shape[0]):
        
        maxval = np.abs(img[i]).max()
        
        img[i] = img[i]/maxval
        
        ksp[i] = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img[i],axes=(-1,-2)), norm='ortho', axes=(-1,-2)), axes=(-1,-2))
    
        us_ksp[i] = ksp[i] * mask[i]
        
        temp = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ksp[i],axes=(-1,-2)), norm='ortho', axes=(-1,-2)), axes=(-1,-2))
    
        org[i] = np.sum(temp*np.conj(csm[i]), axis=0, keepdims=True)
    
    
    
    mask = np.tile(np.expand_dims(mask, axis=1), (1,ncoils,1,1))
    
    org = torch.tensor(org)
    us_ksp = torch.tensor(us_ksp)
    mask = torch.tensor(mask)
    csm = torch.tensor(csm)
    
    
    del temp, ksp, img
    
    
    return org, us_ksp, csm, mask

    
    
    
#%%dataset preparation

    
if __name__ == "__main__":
    
    
    path = 'Data/'
    num_sl = 1
    start_sl = 0
    acc = 6.0
    org_data, us_data, csm_data, mask_data = preprocess_testing_data(path, num_sl, start_sl, acc)
    
    
    dispind = 0
    
    for i in range(0,1):
        print(org_data[i:i+1].numpy().max())
        print(org_data[i:i+1].numpy().min())
        print(us_data[i:i+1].numpy().max())
        print(us_data[i:i+1].numpy().min())
        print('shape of fsim is', org_data[i:i+1].shape)
        print('shape of usk is', us_data[i:i+1].shape)
        print('shape of csm is', csm_data[i:i+1].shape)
        print('shape of mask is', mask_data[i:i+1].shape)
        fsim = org_data[i:i+1].numpy()
        usk = us_data[i:i+1].numpy()
        csm = csm_data[i:i+1].numpy()
        mask = mask_data[i:i+1].numpy()
        usim = np.zeros_like(fsim)
        img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(usk[0], axes=[-1,-2]), axes=(-1,-2), norm="ortho"), axes=[-1,-2])
        usim[0,0] = np.sum(img*np.conj(csm[0]),axis=0).astype(np.complex64)
        fig, axes = plt.subplots(1,3)
        pos = axes[0].imshow(np.abs(fsim[dispind,0,:,:]))
        pos = axes[1].imshow(np.abs(usim[dispind,0,:,:]))
        pos = axes[2].imshow(np.abs(mask[dispind,0,:,:]))
        plt.show()
        break