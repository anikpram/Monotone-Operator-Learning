#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:11:26 2022

@author: apramanik
"""



import torch
import torch.nn as nn
import torch.autograd as autograd


class Unroll(nn.Module):
    def __init__(self,f,K,A_init,lam_init):
        super().__init__()
        self.K=K
        self.f=f
        self.A_init = A_init
        self.lam_init=lam_init
        
    def forward(self, b, csm, mask):

        Atb = self.f.A.adjoint(b, csm)
        
        zero = torch.zeros_like(Atb).to(Atb.device)
        sense_out = self.A_init.inv(zero, self.lam_init*Atb, self.lam_init, csm, mask)
        x = sense_out
        
        for blk in range(self.K):
            xold = x
            x = self.f(x, Atb, csm, mask)
        
        errforward = torch.norm(x-xold)/torch.norm(xold)
        return x, sense_out, errforward, self.K
    
    
    
    
    
class DEQ(nn.Module):
    def __init__(self,f,K,A_init,lam_init,tol=0.05,verbose=True):
        super().__init__()
        self.K=K
        self.f=f
        self.A_init = A_init
        self.lam_init = lam_init
        self.tol = tol
        self.verbose = verbose

    def forward(self, b, csm, mask):

        Atb = self.f.A.adjoint(b, csm)
        zero = torch.zeros_like(Atb).to(Atb.device)
        sense_out = self.A_init.inv(zero, self.lam_init*Atb, self.lam_init, csm, mask)
        x = sense_out
        
        
        with torch.no_grad():         
            
            for blk in range(self.K):
                xold = x
                x=self.f(x, Atb, csm, mask)
                errforward = torch.norm(x-xold)/torch.norm(xold)
                if(self.verbose):
                    print(errforward)
                    print("diff", torch.norm(x-xold).cpu().numpy()," xnew ",torch.norm(x).cpu().numpy()," xold ",torch.norm(xold).cpu().numpy())
                if(errforward < self.tol and blk>2):
                    if(self.verbose):
                        print("exiting front prop after ",blk," iterations with ",errforward )
                    break
                    
        
        z = self.f(x, Atb, csm, mask)  # forward layer seen by pytorch
            
        # For computation of Jacobian vector product
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, Atb, csm, mask)
        
       # Backward propagation of gradients
        def backward_hook(grad):
            g = grad
            for i in range(self.K):
                gold = g
                g = autograd.grad(f0,z0,gold,retain_graph=True)[0] + grad
                errback = torch.norm(g-gold)/torch.norm(gold)
                if(errback < self.tol):
                    if(self.verbose):
                        print("exiting back prop after ",blk," iterations with ",errback )
                    break
            g = autograd.grad(f0,z0,gold)[0] + grad
            #g = torch.clamp(g,min=-1,max=1)
            return(g)

       # Adding the hook to modify the gradients
        z.register_hook(backward_hook)
    
        return z, sense_out, errforward, blk
    
    
class DEQ_inf(nn.Module):
    def __init__(self,f,K,A_init,lam_init,tol=0.05,verbose=True):
        super().__init__()
        self.K=K
        self.f=f
        self.A_init = A_init
        self.lam_init = lam_init
        self.tol = tol
        self.verbose = verbose

    def forward(self, b, csm, mask):

        Atb = self.f.A.adjoint(b, csm)
        zero = torch.zeros_like(Atb).to(Atb.device)
        #import time
        #start = time.time()
        sense_out = self.A_init.inv(zero, self.lam_init*Atb, self.lam_init, csm, mask)
        #end=time.time()
        #print('sense time is:', end-start)
        x = sense_out
        
        
        with torch.no_grad():         
            
            for blk in range(self.K):
                xold = x
                x=self.f(x, Atb, csm, mask)
                errforward = torch.norm(x-xold)/torch.norm(xold)
                if(self.verbose):
                    print(errforward)
                    print("diff", torch.norm(x-xold).cpu().numpy()," xnew ",torch.norm(x).cpu().numpy()," xold ",torch.norm(xold).cpu().numpy())
                if(errforward < self.tol and blk>2):
                    if(self.verbose):
                        print("exiting front prop after ",blk," iterations with ",errforward )
                    break
                    
        
        z = self.f(x, Atb, csm, mask)  # forward layer seen by pytorch
            
    
        return z, sense_out, errforward, blk
    
    


    
    
class Sense_block(nn.Module):
    def __init__(self,A,lam):
        super().__init__()
        self.lam = torch.tensor(lam,dtype=torch.float32)
        self.A = A
        
    def forward(self, b, csm, mask):
        
        Atb = self.A.adjoint(b, csm)
        zero = torch.zeros_like(Atb).to(Atb.device)
        xsense = self.A.inv(zero, self.lam*Atb, self.lam, csm, mask)
        x = xsense
        xold = x
        
        errforward = torch.norm(x-xold)/torch.norm(xold)
        return x, xsense, errforward, self.lam
    
    
    
    
    
    

    
    
    
#----------------------------
# Unroll admm
class Unroll_admm(nn.Module):
    def __init__(self,f,K,A_init,lam_init):
        super().__init__()
        self.K=K
        self.f=f
        self.A_init = A_init
        self.lam_init=lam_init
        
    def forward(self, b, csm, mask):
        Atb = self.A_init.adjoint(b, csm)
        zero = torch.zeros_like(Atb).to(Atb.device)
        #x = self.f.A.inv(self.f.lam*Atb, self.f.lam*Atb, self.f.lam, csm, mask)
        sense_out = self.A_init.inv(zero, self.lam_init*Atb, self.lam_init, csm, mask)
        x = sense_out
        u = zero
        xold = x
        for blk in range(self.K):
            xold = x
            x,u=self.f(x,u,Atb,csm,mask)
        
        errforward = torch.norm(x-xold)/torch.norm(xold)
        return x, sense_out, errforward, self.K
    
    
#---------------------------------------
# Direct Inversion
class Direct_inversion(nn.Module):
    def __init__(self,f,K,A_init,lam_init):
        super().__init__()
        self.K=K
        self.f=f
        self.A_init = A_init
        self.lam_init=lam_init
        
    def forward(self, b, csm, mask):
        Atb = self.A_init.adjoint(b, csm)
        zero = torch.zeros_like(Atb).to(Atb.device)
        #x = self.f.A.inv(self.f.lam*Atb, self.f.lam*Atb, self.f.lam, csm, mask)
        sense_out = self.A_init.inv(zero, self.lam_init*Atb, self.lam_init, csm, mask)
        x = sense_out
        #x_adj = self.A_init.adjoint(x, csm)
        #xold = x_adj
        xold = x
        x = self.f(x)
        
        #x = self.A_init.adjoint(x, csm)
        
        errforward = torch.norm(x-xold)/torch.norm(xold)
        return x, sense_out, errforward, self.K
    
    
    
    
    
    
    
    