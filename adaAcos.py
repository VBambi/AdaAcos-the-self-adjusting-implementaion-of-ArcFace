# File:     adaAcos.py                              #
# Author:   Vincent Bamberger                       #
# Date:     16.04.2024                              #
# ------------------------------------------------- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaAcos(nn.Module):
    def __init__(self, num_features, num_classes):
        super(AdaAcos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = np.log(num_classes-1) / np.cos(np.pi/2.3) #This is inspired by fixed AdaCos, though theta_y is changed from pi/4: https://arxiv.org/abs/1905.00292
        print(f"autoscale s set to {self.s}")
        self.m = 0
        
        self.W = nn.Parameter(torch.FloatTensor(num_features, num_classes))
        nn.init.xavier_uniform_(self.W) # random weight initialisation

    def forward(self, x, y=None):
        # normalize features
        x = F.normalize(x, 2, 1)
        # normalize weights
        W = F.normalize(self.W, 2, 0)
        # dot product
        logits = x@W
        
        if y is not None:
            with torch.no_grad():
                theta = torch.acos(torch.clamp(logits, -1.0+1e-7, 1.0-1e-7))
                theta_false_min = torch.min(theta[y!=1].view(theta.shape[0],-1),axis=1)[0]  # angle of closest uncorresponding class
                m = torch.clamp(torch.median(theta_false_min-theta[y==1]), min=0)           # compute m = difference of theta_y and theta_false_min
                self.m = m.item() # to easily view or log m during training
                
                logits[y==1] = torch.cos(torch.clamp(theta[y==1]+m, max=np.pi))             # penalize theta_y with m
        
        return self.s * logits
