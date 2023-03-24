#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:19:55 2022

@author: apramanik
"""



import torch
import torch.nn as nn



class cg_block(nn.Module):
    def __init__(self, cgIter, cgTol):
        super(cg_block, self).__init__()
        self.cgIter = cgIter
        self.cgTol = cgTol
        
    def forward(self, lhs, rhs, x0):
        fn=lambda a,b: torch.abs(torch.sum(torch.conj(a)*b,axis=[-1,-2,-3]))
        x = x0
        r = rhs-lhs(x0)
        p = r
        rTr = fn(r,r)
        eps=torch.tensor(1e-10)
        for i in range(self.cgIter):
            Ap = lhs(p)
            alpha=rTr/(fn(p,Ap)+eps)
            x = x +  alpha[:,None,None,None] * p
            r = r -  alpha[:,None,None,None] * Ap
            rTrNew = fn(r,r)
            if torch.sum(torch.sqrt(rTrNew+eps)) < self.cgTol:
                break
            beta = rTrNew / (rTr+eps)
            p = r + beta[:,None,None,None] * p
            rTr=rTrNew
           
        return x