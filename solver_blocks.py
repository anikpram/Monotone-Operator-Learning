#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:10:37 2022

@author: apramanik
"""



import torch
import torch.nn as nn
from networks import dwblock


class modlBlock(nn.Module):
    def __init__(self, A, lam, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super(modlBlock, self).__init__()
        self.dw = dwblock(input_channels, features, output_channels, number_of_layers, spectral_norm)
        self.lam = nn.Parameter(torch.tensor(lam,dtype=torch.float32))
        self.A = A
        self.alpha = torch.tensor(1.0,dtype=torch.float32)
    
        
    def forward(self, x, Atb, csm, mask):
        z = x - self.dw(x)
        rhs = z + self.lam*Atb
        x = self.A.inv(x, rhs, self.lam, csm, mask)
        return x
    
    
    
    
class fwdbwdBlock(nn.Module):
    def __init__(self, A, lam, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super(fwdbwdBlock, self).__init__()
        self.dw = dwblock(input_channels, features, output_channels, number_of_layers, spectral_norm)
        self.lam = nn.Parameter(torch.tensor(lam,dtype=torch.float32))
        self.A = A
        self.alpha = torch.tensor(0.1,dtype=torch.float32)
    
        
    def forward(self, x, Atb, csm, mask):
        z = self.dw(x)
        rhs = (1 - self.alpha)*x + self.alpha*z + self.alpha*self.lam*Atb
        x = self.A.inv(x, rhs, self.alpha*self.lam, csm, mask)
        return x
    
    
    
    

class gradBlock(nn.Module):
    def __init__(self, A, lam, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super(gradBlock, self).__init__()
        
        
        self.dw = dwblock(input_channels, features, output_channels, number_of_layers, spectral_norm)
        #self.lam = nn.Parameter(torch.tensor(lam,dtype=torch.float32))
        self.lam = torch.tensor(lam,dtype=torch.float32)
        self.A = A
        self.alpha = torch.tensor(1.0,dtype=torch.float32)
        
        
    
        
    def forward(self, x, Atb, csm, mask):
        #z = x - self.dw(x)
        z = self.dw(x)
        
        
        #x = x + 1e-5*self.lam*((Atb - self.A.adjoint(self.A.forward(x, csm, mask), csm)) - z)
        
        x = x - self.lam*((self.A.adjoint(self.A.forward(x, csm, mask), csm) - Atb) + z)
        return x
        #return z
        
        
        
    
    
class molgradBlock(nn.Module):
    def __init__(self, A, lam, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super(molgradBlock, self).__init__()
        
        
        self.dw = dwblock(input_channels, features, output_channels, number_of_layers, spectral_norm)
        self.lam = nn.Parameter(torch.tensor(lam,dtype=torch.float32))
        #self.lam = torch.tensor(lam,dtype=torch.float32)
        self.A = A
        self.alpha = torch.tensor(1.0,dtype=torch.float32)
        
        
    
        
    def forward(self, x, Atb, csm, mask):
        z = self.dw(x)
        dcg = self.lam*(self.A.ATA(x, csm, mask) - Atb)
        x = (1 - self.alpha)*x + self.alpha*(z - dcg) 
        return x
    
    
    





class admmBlock(nn.Module):
    def __init__(self, A, lam, input_channels, features, output_channels, number_of_layers, spectral_norm=False):
        super(admmBlock, self).__init__()
        self.dw = dwblock(input_channels, features, output_channels, number_of_layers, spectral_norm)
        self.lam = nn.Parameter(torch.tensor(lam,dtype=torch.float32))
        #self.lam = torch.tensor(lam,dtype=torch.float32)
        self.A = A
        self.alpha = torch.tensor(1.0,dtype=torch.float32)
    
        
    def forward(self, x, u, Atb, csm, mask):
        
        z = x - u - self.dw(x - u)
        rhs = z + self.lam*Atb + u
        x = self.A.inv(x, rhs, self.lam, csm, mask)
        u = u + z - x 
        return x, u
    
    
    
    


