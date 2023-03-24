#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:18:59 2022

@author: apramanik
"""


import torch
import torch.fft
import torch.nn as nn
from cg import cg_block



class sense(nn.Module):
    def __init__(self, cgIter):
        super().__init__()
        
        self.cgIter = cgIter
        self.cg = cg_block(self.cgIter, 1e-9)
        

    def forward(self, img, csm, mask):
        cimg = img*csm
        mcksp = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(cimg, dim=[-1,-2]), dim=[-1,-2], norm="ortho"), dim=[-1,-2])
        usksp = mcksp * mask
        return usksp
        
    def adjoint(self, ksp, csm):
        img = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
        cs_weighted_img = torch.sum(img*torch.conj(csm),1,True)
        return cs_weighted_img
    
    
    def ATA(self, img, csm, mask):
        cimg = img*csm
        mcksp = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(cimg, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
        usksp = mcksp * mask
        usimg = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(usksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
        cs_weighted_img = torch.sum(usimg*torch.conj(csm),1,True)
        return cs_weighted_img
    
    
    def inv(self, x0, rhs, lam, csm, mask):
        
        lhs = lambda x: lam*self.ATA(x, csm, mask) + 1.001*x
        out = self.cg(lhs, rhs, x0)
        
        return out