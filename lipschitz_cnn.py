#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:09:50 2022

@author: apramanik
"""



import torch
import torch.nn as nn



def l2_norm(x):
    return torch.norm(x.view(x.shape[0], -1), p=2, dim=1)


class Lipschitz(nn.Module):
    def __init__(self, u, eps, shap, model,lr=1e8):
        super().__init__()
        self.shap = shap
        self.model = model
        self.lr = lr
        self.eps = eps
        self.u = u
        self.gpu=torch.device('cuda')
        
        self.u = self.u.to(self.gpu)
        self.eps = self.eps.to(self.gpu)
        self.v = torch.complex(torch.rand(self.shap,dtype=torch.float32),torch.rand(self.shap,dtype=torch.float32)) 
        self.v = self.v.to(self.gpu)
        self.v = self.u + self.eps*self.v 
        
        
        self.v = self.v.requires_grad_(True)  

        
    def compute_ratio(self):
        u_out = self.model(self.u)
        v_out = self.model(self.v)
        loss = l2_norm(u_out - v_out)
        loss = loss/l2_norm(self.u - self.v)
        return loss

    def adverserial_update(self, iters=1,reinit=False):
        
        if(reinit):
            self.v = torch.complex(torch.rand(self.shap,dtype=torch.float32),torch.rand(self.shap,dtype=torch.float32)) 
            self.v = self.v.to(self.gpu)
            self.v = self.u + self.eps*self.v 
        
        self.v = self.v.requires_grad_(True) 
        
        for i in range(iters):
            loss = self.compute_ratio()
            loss_sum = torch.sum(loss)
            loss_sum.backward()

            
            v_grad = self.v.grad.detach()
            v_tmp = self.v.data + self.lr * v_grad
            v_tmp = (v_tmp/torch.norm(v_tmp))*torch.norm(self.u)


            self.v.grad.zero_()

            self.v.data = v_tmp
        
        self.v = self.v.requires_grad_(False)  
    
        loss_sum = self.compute_ratio()
        return loss_sum