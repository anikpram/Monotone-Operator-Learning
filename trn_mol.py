#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:13:41 2023

@author: apramanik
"""




from collections import OrderedDict
import yaml
from torch import optim
import numpy as np
import os, torch, time
import torch.nn as nn
from tqdm import tqdm
import random
from scipy.io import savemat

from Sense import sense
from lipschitz_cnn import Lipschitz
from solver_blocks import fwdbwdBlock
from solvers import DEQ
from data_preprocessing import preprocess_training_data





#%%

def contdifflog(x, T, e):
    
    condition = (x < (T - e))*torch.tensor(1.0)
    
    y1 = -torch.log((T - x)*condition + (1.0 - condition))
    
    y2 = -torch.log(e) + (1/e)*(x - T + e)
    
    return y1*condition + y2*(1.0 - condition)









#%%


def read_yaml(file):
    """ A function to read YAML file"""
    with open(file) as f:
        config = list(yaml.safe_load_all(f))
        
    return config


def write_yaml(data, path):
    """ A function to write YAML file"""
    with open(path, 'w') as f:
        yaml.dump(data, f)


def save_checkpoint(states,  path, filename='model_best.pth.tar'):
    checkpoint_name = os.path.join(path,  filename)
    torch.save(states, checkpoint_name)







#%%


training_epochs = 100
learning_rate = 1e-3
learning_rate_lam = 1 
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
learning_rate_lipschitz = 1e0
clip_wt = 1e-6
eps = 0.01
alpha = 0.01
T = 0.98
e = 1e-3


restore = False
restore_dir = 'Fastmri_1_Sub_1_Slices_MoDL_1itr'
restore_model  = 'model-1000.pth.tar'

save_model_every_N_epochs = 20


num_sl = 360
start_sl = 0
acc = 6.0
path = 'Data/'




od = OrderedDict()

od["training_epochs"] = training_epochs
od["learning_rate"] = learning_rate
od["learning_rate_lam"] = learning_rate_lam
od["number_of_feature_filters"] = number_of_feature_filters
od["number_of_layers"] = number_of_layers


od["input_channels"] = input_channels
od["output_channels"] = output_channels
od["lam_itr"] = lam_itr
od["lam_init"] = lam_init
od["cgIter_itr"] = cgIter_itr
od["cgIter_init"] = cgIter_init
od["learning_rate_lipschitz"] = learning_rate_lipschitz
od["clip_wt"] = clip_wt
od["eps"] = eps
od["alpha"] = alpha
od["lip_thresh"] = T
od["lip_eps"] = e


od["restore"] = restore
od["restore_dir"] = restore_dir
od["restore_model"] = restore_model

od["save_model_every_N_epochs"] = save_model_every_N_epochs


od["num_sl"] = num_sl
od["start_sl"] = start_sl
od["acc"] = acc
od["path"] = path






#%%


### Define the model, optimizer and loss functions


saveDir = 'Models/'   
directory = saveDir + 'UIHC_brain_' + str(num_sl) + '_Slices_MOL_trial'




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A_init = sense(cgIter_init).to(device)
A = sense(cgIter_itr).to(device)


optBlock = fwdbwdBlock(A, lam_itr, input_channels, number_of_feature_filters, output_channels, number_of_layers, spectral_norm=False)
optBlock.alpha = torch.tensor(alpha,dtype=torch.float32)

  
model = DEQ(optBlock, nFETarget, A_init, lam_init, tol=err_tol, verbose=False)
model = model.to(device)

if restore:
    restore_dir = saveDir + restore_dir
    model.load_state_dict(torch.load(os.path.join(restore_dir, restore_model))['state_dict'])

loss_function = nn.MSELoss()

optimizer = optim.Adam([
            {'params': model.f.dw.parameters(), 'lr': learning_rate},
            {'params': model.f.lam, 'lr': learning_rate_lam}
        ])
    
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.3)





if not os.path.exists(directory):
    os.makedirs(directory)
    
best_training_loss = 1e10
training_steps_loss, lip_pred_steps, err_steps, nFE_steps, clip_steps_loss, mse_steps_loss = [], [], [], [], [], []
training_epochs_loss, lip_pred_epochs, err_epochs, nFE_epochs, clip_epochs_loss, mse_epochs_loss = [], [], [], [], [], []





yaml_path_parameters = directory + '/training_parameters.yml'
write_yaml(od, yaml_path_parameters)
   



#%% Load Data


org_data, us_data, csm_data, mask_data = preprocess_training_data(path, num_sl, start_sl, acc)



#%%


### Train 

T = torch.tensor(T)

e = torch.tensor(e)

eps = torch.as_tensor(eps)


for epoch in tqdm(range(1,training_epochs+1)):
        
    start_time = time.time()
    model.train()
    
    indices = np.random.permutation(org_data.shape[0])
    
    steps_ep = 0
    
    for i in indices:
        
        steps_ep = steps_ep + 1
    
        target_fully_sampled = org_data[i:i+1].to(device)
        target_fully_sampled = torch.abs(target_fully_sampled)
        coil_sensitivity_maps = csm_data[i:i+1].to(device)
        input_under_sampled = us_data[i:i+1].to(device)
        mask = mask_data[i:i+1].to(device)
        
        
        predicted_fully_sampled, sense_out, err, nFE  = model(input_under_sampled, coil_sensitivity_maps, mask)
        
        u = torch.clone(predicted_fully_sampled).detach()
        lip_pred = Lipschitz(u, eps, u.shape, optBlock.dw, lr=learning_rate_lipschitz)
        lip_pred_est = lip_pred.adverserial_update(iters=10)
        
        
        predicted_fully_sampled = torch.abs(predicted_fully_sampled)
        
        
        mse_loss = loss_function(predicted_fully_sampled, target_fully_sampled)
        
        
        clip_loss = clip_wt*contdifflog(lip_pred_est, T, e)
    
    
    
        loss = mse_loss + clip_loss 
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_steps_loss.append(loss.detach().cpu().numpy())
        lip_pred_steps.append(lip_pred_est.detach().cpu().numpy()[0])
        err_steps.append(err.detach().cpu().numpy())
        nFE_steps.append(nFE)
        clip_steps_loss.append(clip_loss.detach().cpu().numpy())
        mse_steps_loss.append(mse_loss.detach().cpu().numpy())
        
        
        del lip_pred_est
        torch.cuda.empty_cache()
    
    
    training_epochs_loss.append(np.sum(training_steps_loss[(epoch-1)*steps_ep: epoch*steps_ep]))
    lip_pred_epochs.append(np.mean(lip_pred_steps[(epoch-1)*steps_ep: epoch*steps_ep]))
    nFE_epochs.append(np.mean(nFE_steps[(epoch-1)*steps_ep: epoch*steps_ep]))
    err_epochs.append(np.mean(err_steps[(epoch-1)*steps_ep: epoch*steps_ep]))
    clip_epochs_loss.append(np.sum(clip_steps_loss[(epoch-1)*steps_ep: epoch*steps_ep]))
    mse_epochs_loss.append(np.sum(mse_steps_loss[(epoch-1)*steps_ep: epoch*steps_ep]))
    
    
    print('Epoch:%d, tloss: %.6f, mloss: %.6f, closs: %.6f, lip_pred: %.3f, alpha: %.3f, lam: %.3f, nFE %d, error %.5f' % (epoch, training_epochs_loss[epoch-1], mse_epochs_loss[epoch-1], clip_epochs_loss[epoch-1], lip_pred_epochs[epoch-1], optBlock.alpha.detach().cpu().numpy(), optBlock.lam.detach().cpu().numpy(), nFE_epochs[epoch-1], err_epochs[epoch-1]))
    
    if mse_epochs_loss[epoch-1] > 2*best_training_loss:
        scheduler.step()
    
    
    save_checkpoint(
            {
        
        'state_dict': model.state_dict(),
        
    },
    path=directory,
    filename='model_ep.pth.tar'
            )
    
    
    
    if np.remainder(epoch, save_model_every_N_epochs)==0:
        save_checkpoint(
            {
        
        'state_dict': model.state_dict()
        
    },
    path=directory,
    filename='model-{}.pth.tar'.format(epoch)
            )
    
    
        
    if mse_epochs_loss[epoch-1] < best_training_loss:    
        best_training_loss = mse_epochs_loss[epoch-1]
        
        save_checkpoint(
    {
        'state_dict': model.state_dict(),
    },
    path=directory,
    filename='model_best_trn.pth.tar'
    )
      
        
    savemat(directory+'/training_steps_loss.mat',mdict={'steps':training_steps_loss},appendmat=True)
    savemat(directory+'/lip_pred_steps.mat',mdict={'steps':lip_pred_steps},appendmat=True)
    savemat(directory+'/err_steps.mat',mdict={'steps':err_steps},appendmat=True)
    savemat(directory+'/nFE_steps.mat',mdict={'steps':nFE_steps},appendmat=True)
    savemat(directory+'/clip_steps_loss.mat',mdict={'steps':clip_steps_loss},appendmat=True)
    savemat(directory+'/mse_steps_loss.mat',mdict={'steps':mse_steps_loss},appendmat=True)
    
    savemat(directory+'/training_epochs_loss.mat',mdict={'epochs':training_epochs_loss},appendmat=True)
    savemat(directory+'/lip_pred_epochs.mat',mdict={'epochs':lip_pred_epochs},appendmat=True)
    savemat(directory+'/err_epochs.mat',mdict={'epochs':err_epochs},appendmat=True)
    savemat(directory+'/nFE_epochs.mat',mdict={'epochs':nFE_epochs},appendmat=True)
    savemat(directory+'/clip_epochs_loss.mat',mdict={'epochs':clip_epochs_loss},appendmat=True)
    savemat(directory+'/mse_epochs_loss.mat',mdict={'epochs':mse_epochs_loss},appendmat=True)
        
        










    
    
    
    


















