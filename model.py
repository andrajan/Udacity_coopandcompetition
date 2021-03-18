import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    def __init__(self, state_size, action_size, seed, param):

        super(Actor, self).__init__()
        self.layer_units=param['layer_units']
        
        
        self.hlayers=buildFnetwork(self.layer_units,state_size)
        self.out=nn.Linear(self.layer_units[-1],action_size)
        reset_parameters(self.out,final=True)
        self.outlayer=nn.Sequential(self.out,nn.Tanh())
        

    def forward(self, state):

        x = self.hlayers(state)
        return self.outlayer(x)
    


class Critic(nn.Module):

    def __init__(self, state_size ,action_size,seed,param):


        super(Critic, self).__init__()
        self.layer_units=param['layer_units']
        self.type=param['ddpgtype']
         
        self.hlayers=buildFnetwork(self.layer_units,state_size+action_size)
        
        if self.type=='catdist' or self.type=='sampleNdist':
            atoms=param['atoms']
            self.out=nn.Linear(self.layer_units[-1],atoms)
            reset_parameters(self.out,final=True)
            self.outlayer=nn.Sequential(self.out,nn.Softmax(dim=-1))
        if self.type=='singleval':
            self.outlayer=nn.Linear(self.layer_units[-1],1)
            reset_parameters(self.outlayer,final=True)

        

    def forward(self,intuple):
        if self.type=='singleval' or self.type=='catdist':
            x=torch.cat(intuple,dim=-1)
        if self.type=='sampleNdist':
            x=torch.cat(intuple,dim=-1)
        x=self.hlayers(x)
        return self.outlayer(x)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

def buildFnetwork(layer_units,insize):
    #builds our hidden layers based upon input parameters
    
    layers=[]
    l_in=insize
    for i in range(len(layer_units)):
        l_out=layer_units[i]
        layer=[nn.Linear(l_in,l_out), nn.SELU()]
        reset_parameters(layer[0])
        layers+=layer
        l_in=l_out
    return nn.Sequential(*layers)
    

def reset_parameters(layer,final=False):
    #resets parameters to initialized values
    
    if final:
        layer.weight.data.uniform_(-3e-3, 3e-3)
    else:
        layer.weight.data.uniform_(*hidden_init(layer))




