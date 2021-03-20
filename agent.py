import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Actor,Critic
from torch.distributions.normal import Normal
import torch.nn.functional as F
from utils import OUNoise


class Agent():
    
    def __init__(self,param):
        
        self.device=param['device']
        self.action_size=param['action_size']
        self.state_size=param['state_size']
        self.seed=param['seed']
        self.update_times=param['update_times']
        printyn=param['print']
        
        self.ddpgtype=param['ddpgtype']
        self.atoms=param['atoms']
        
        self.criticparam=param['criticparam']
        self.criticparam['ddpgtype']=param['ddpgtype']
        self.criticparam['atoms']=self.atoms
        self.actorparam=param['actorparam']
        

        if self.ddpgtype=='catdist':
            self.criticparam['atoms']=self.atoms
            self.catparam=param['catparam']
            self.vmin=self.catparam['vmin']
            self.vmax=self.catparam['vmax']
            self.qspace=torch.linspace(self.vmin,self.vmax,self.atoms).type(torch.FloatTensor).to(self.device)
        
        self.gamma=param['gamma']
        
        self.lr_actor=param['lr_actor']
        self.lr_critic=param['lr_critic']
        self.l2weights=param['l2weights']
        
        self.tau=param['tau']
        
        self.batch_size=param['batch_size']
        self.replay_param=param['replay_param']
        self.replay_param['batch_size']=self.batch_size
        
        self.OUparam=param['OUparam']
        self.OUparam['seed']=self.seed
        
        self.criticparam=param['criticparam']
        self.criticparam['ddpgtype']=param['ddpgtype']
        self.actorparam=param['actorparam']

        #creates our actor and critic
        self.actionestimator_local=Actor(self.state_size,self.action_size,self.seed,self.actorparam).to(self.device)
        self.actionestimator_target=Actor(self.state_size,self.action_size,self.seed,self.actorparam).to(self.device)
        self.action_optimizer=optim.Adam(self.actionestimator_local.parameters(),lr=self.lr_actor)
        
        self.Qval_local=Critic(self.state_size*2,self.action_size,self.seed,self.criticparam).to(self.device)
        self.Qval_target=Critic(self.state_size*2,self.action_size,self.seed,self.criticparam).to(self.device)
                
        self.Q_optimizer=optim.Adam(self.Qval_local.parameters(),lr=self.lr_critic,weight_decay=self.l2weights)
        
        if printyn:
            print(self.Qval_local)
            print(self.actionestimator_local)

        #Here we create OU noise for each of our pnum parameters in our actor model.
        pnum=sum(p.numel() for p in self.actionestimator_local.parameters())
        self.noise=OUNoise(pnum,self.OUparam)
        
        self.samplenonise=torch.ones(self.atoms)
        self.stepnum=0
        

        
    def act(self,state,noise):
        
        
        state=torch.from_numpy(state).float().to(self.device)
        self.actionestimator_local.eval()
        # if noise we add OU noise to each parameter in order to promote exploration early in the training process
        if noise:
            for actionp,noisep in zip (self.actionestimator_local.parameters(),self.noise.sample()):
                actionp=actionp+noisep
            
        with torch.no_grad():
            action=self.actionestimator_local(state).to('cpu').numpy()
            
        self.actionestimator_local.train()
        
        action=np.clip(action,-1,1)
        
        return action
    
            
    def criticloss(self,state,action,reward,next_state,next_states,done,n_step):
        qinput=(state,action)
        Qests=self.Qval_local(qinput)
        
        with torch.no_grad():
            next_action=self.actionestimator_target(next_state)
            qinput_target=(torch.Tensor(next_states.view(self.batch_size,-1)),torch.Tensor(next_action))
            Qtargets=self.Qval_target(qinput_target)*(1-done)

        
        #select Qloss based on what DDPG or D4PG
        if self.ddpgtype=='catdist':
            Qloss=self.crossentropy_loss(Qtargets,Qests,reward,n_step)
            
        elif self.ddpgtype=='singleval':
            Qloss=F.mse_loss(reward+self.gamma**n_step*Qtargets,Qests)
                        
        return Qloss

    def crossentropy_loss(self,probs_target,probs_est,reward,n_step):
    
        delta_z=(self.vmax-self.vmin)/(self.atoms-1)
        tz=torch.clamp(reward+self.qspace*self.gamma**n_step,self.vmin,self.vmax)
        bj=(tz-self.vmin)/delta_z
        l=torch.floor(bj).view(probs_target.shape[0],-1)
        up=torch.ceil(bj).view(probs_target.shape[0],-1)
        ml=(probs_target*(bj+(l==up).float()-l)).view(*l.shape)
        mup=(probs_target*(up-bj)).view(*l.shape)
        
        #empty array is set up and then filled with elements of ml and mup based on mask. See d4pg paper for details
        #about loss function and why we need to project the expected q.
        m=torch.zeros(*l.shape).to(self.device)
        maskl=l.type(torch.LongTensor).view(*l.shape).to(self.device)
        maskup=up.type(torch.LongTensor).view(*l.shape).to(self.device)
        
        for i in range(probs_target.shape[0]):
            m[i].index_add_(0,maskl[i],ml[i]).index_add_(0,maskup[i],mup[i])

        

        loss=-torch.sum(m*torch.log(probs_est+1e-30),dim=-1)
        return loss

    
    def expectedQ(self,critic_output):
        
        #gives the expected q value based on distribution
        if self.ddpgtype=='catdist':
            Q=torch.sum(self.qspace*critic_output,dim=-1)
        elif self.ddpgtype=='singleval' or self.ddpgtype=='sampleNdist':
            Q=critic_output
            
        return Q 
        


    def softupdate(self,target,local):
    
        for targetp,localp in zip(target.parameters(),local.parameters()):
            targetp.data.copy_(self.tau*localp.data+(1-self.tau)*targetp.data)

            targetp.data.copy_(self.tau*localp.data+(1-self.tau)*targetp.data)

