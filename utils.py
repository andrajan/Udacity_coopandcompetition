import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch


class ReplayBuffer:

    def __init__(self,device,param):
        
        self.device=device
        self.batch_size=param['batch_size']
        self.buffer_size=param['buffer_size']

        self.memory=deque(maxlen=self.buffer_size)
        self.experience=namedtuple('Experience',field_names=['states','actions','rewards','next_states','dones'])
        
    def add(self,state,action,reward,next_state,done):
        self.memory.append(self.experience(state,action,reward,next_state,done))
        
        
    def sample(self):
        sampler=random.choices(self.memory,k=self.batch_size)
        
        states=torch.from_numpy(np.vstack([[exp.states for exp in sampler if exp is not None]])).float().to(self.device)
        rewards=torch.from_numpy(np.vstack([[exp.rewards for exp in sampler if exp is not None]])).float().to(self.device)
        next_states=torch.from_numpy(np.vstack([[exp.next_states for exp in sampler if exp is not None]])).float().to(self.device)
        actions=torch.from_numpy(np.vstack([[exp.actions for exp in sampler if exp is not None]])).float().to(self.device)
        dones=torch.from_numpy(np.vstack([[exp.dones for exp in sampler if exp is not None]]).astype(np.uint8)).float().to(self.device)
        return states,rewards,next_states,actions,dones
    
    def lenmemory(self):
        return len(self.memory)


class OUNoise:

    def __init__(self, size, param):
        self.mu = param['mu'] * np.ones(size)
        self.theta = param['theta']
        self.sigma = param['sigma']
        self.seed = random.seed(param['seed'])
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
