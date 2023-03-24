#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 16:33:14 2022

@author: apramanik
"""





from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import os, torch, time
import torch.nn as nn
from tqdm import tqdm
import random
from scipy.io import savemat

from Sense import sense
from lipschitz_cnn import Lipschitz
from solver_blocks import fwdbwdBlock
from solvers import DEQ_inf
from data_preprocessing import preprocess_testing_data, preprocess_training_data











#%%



number_of_feature_filters = 64 
number_of_layers = 5  


input_channels = 1
output_channels = 1
lam_itr = 100
lam_init = 100
cgIter_itr = 5
cgIter_init = 50
err_tol = 1e-4
nFETarget = 500
alpha = 0.01



num_sl = 90
start_sl = 0
acc = 6.0
path = 'Data/'



restore_dir = 'UIHC_brain_360_Slices_MOL'
restore_model  = 'model_best_trn.pth.tar'




#%%


### Define the model, optimizer and loss functions

    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A_init = sense(cgIter_init).to(device)
A = sense(cgIter_itr).to(device)


optBlock = fwdbwdBlock(A, lam_itr, input_channels, number_of_feature_filters, output_channels, number_of_layers, spectral_norm=False)
optBlock.alpha = torch.tensor(alpha,dtype=torch.float32)

  
model = DEQ_inf(optBlock, nFETarget, A_init, lam_init, tol=err_tol, verbose=False)
model = model.to(device)


mainDir = 'Models/'  
restore_dir = mainDir + restore_dir
model.load_state_dict(torch.load(os.path.join(restore_dir, restore_model))['state_dict'])





#%%

#org_data, us_data, csm_data, mask_data = preprocess_training_data(path, num_sl, start_sl, acc)
org_data, us_data, csm_data, mask_data = preprocess_testing_data(path, num_sl, start_sl, acc)



#%%

total_sl = org_data.shape[0]


prediction = np.zeros((total_sl,) + us_data.shape[-2:],dtype=np.float32)
prediction_sense = np.zeros((total_sl,) + us_data.shape[-2:],dtype=np.float32)
target = np.zeros((total_sl,) + us_data.shape[-2:],dtype=np.float32)
inputs = np.zeros((total_sl,) + us_data.shape[-2:],dtype=np.float32)
psnri = np.zeros(total_sl,)
psnrf = np.zeros(total_sl,)
ssimi = np.zeros(total_sl,)
ssimf = np.zeros(total_sl,)
err_slices = np.zeros(total_sl,)
nFE_slices = np.zeros(total_sl,)




for i in range(total_sl):
    
    target_fully_sampled = org_data[i:i+1].to(device)
    target_fully_sampled = torch.abs(target_fully_sampled)
    coil_sensitivity_maps = csm_data[i:i+1].to(device)
    input_under_sampled = us_data[i:i+1].to(device)
    #input_under_sampled = A.adjoint(input_under_sampled, coil_sensitivity_maps)
    mask = mask_data[i:i+1].to(device)
    #atb = A.adjoint(input_under_sampled, coil_sensitivity_maps).to(device)
        
    
    with torch.no_grad():    
        predicted_fully_sampled, predicted_sense, err, nFE  = model(input_under_sampled, coil_sensitivity_maps, mask)

    
    input_under_sampled = A.adjoint(input_under_sampled, coil_sensitivity_maps)

    prediction[i] = np.squeeze(torch.abs(predicted_fully_sampled).detach().cpu().numpy())
    target[i] = np.squeeze(target_fully_sampled.detach().cpu().numpy())
    inputs[i] = np.squeeze(torch.abs(input_under_sampled).detach().cpu().numpy())
    prediction_sense[i] = np.squeeze(torch.abs(predicted_sense).detach().cpu().numpy())
    err_slices[i] = err.detach().cpu().numpy()
    nFE_slices[i] = nFE
    
    
    #psnri[i] = peak_signal_noise_ratio(target[i], prediction_sense[i], data_range=target[i].max()) 
    psnri[i] = peak_signal_noise_ratio(target[i], inputs[i], data_range=target[i].max()) 
    psnrf[i] = peak_signal_noise_ratio(target[i], prediction[i], data_range=target[i].max()) 
    
    
    #ssimi[i] = structural_similarity(target[i], prediction_sense[i])
    ssimi[i] = structural_similarity(target[i], inputs[i])
    ssimf[i] = structural_similarity(target[i], prediction[i])
    
    #error_img = np.abs(target[i] - prediction[i])
    
    
    
    
print('Mean nFE', np.mean(nFE_slices).round(3))
print('Mean error', np.mean(err_slices).round(3))
print('Mean SENSE PSNR:', np.mean(psnri).round(3))
print('Mean MoDL PSNR:', np.mean(psnrf).round(3))            
print('Mean SENSE SSIM:', np.mean(ssimi).round(3))
print('Mean MoDL SSIM:', np.mean(ssimf).round(3))  



del us_data, org_data, csm_data, mask_data

error = np.abs(target - prediction)
error_cs = np.abs(target - prediction_sense)
    
#%% Display the output images
plot= lambda x: plt.imshow(x,interpolation='bilinear',cmap=plt.cm.gray, vmax=0.8)
ploterr= lambda x: plt.imshow(x,interpolation='bilinear',cmap=plt.cm.gray,vmax=0.3)
stco=0
endco=-1
stro=0
endro=-1
dispind = 60
rot_fact = 0
plt.clf()
plt.subplot(141)
plot(np.rot90(target[dispind,stco:endco,stro:endro], rot_fact))
plt.axis('off')
plt.title('Fully Sampled \n Ground truth')
plt.subplot(142)
#plot(np.rot90(prediction_sense[dispind,stco:endco,stro:endro], rot_fact))
plot(np.rot90(inputs[dispind,stco:endco,stro:endro], rot_fact))
plt.title('SENSE, PSNR='+ str(psnri[dispind].round(2)) +' dB \n SSIM=' + str(ssimi[dispind].round(3)))
plt.axis('off')
plt.subplot(143)
plot(np.rot90(prediction[dispind,stco:endco,stro:endro], rot_fact))
plt.title('MOL, PSNR='+ str(psnrf[dispind].round(2)) +' dB \n SSIM=' + str(ssimf[dispind].round(3)) + ', Lam=' + str(optBlock.lam.detach().cpu().numpy().round(2)))
plt.axis('off')
plt.subplot(144)
ploterr(np.rot90(error[dispind,stco:endco,stro:endro], rot_fact))
plt.title('Error')
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
plt.show()










