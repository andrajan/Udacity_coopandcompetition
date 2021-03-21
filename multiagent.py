from agent import Agent
from utils import ReplayBuffer
import torch
import numpy as np


class MultiAgent():

    def __init__(self,param):
        super(MultiAgent, self).__init__()

        self.update_times=param['update_times']

        
        self.replay_param=param['replay_param']
        self.batch_size=param['batch_size']
        self.replay_param['batch_size']=self.batch_size

        self.device=param['device']
        param['agent_num']=2
        self.iter=0

        self.multiagent=[Agent(param),Agent(param)]
        self.memory=ReplayBuffer(self.device,self.replay_param)

    def act(self,total_states):
        total_states=[torch.tensor(state,dtype=torch.float) for state in total_states]
        actions=[agent.actionestimator_local(state) for state,agent in zip(total_states,self.multiagent)]
        return actions
    
    def step(self,state,action,reward,next_state,done,n_step,logger):
        done=done.reshape(-1,1)
        reward=reward.reshape(-1,1)
        self.memory.add(state,action,reward,next_state,done)

        if self.batch_size<self.memory.lenmemory():
            for i in range(self.update_times):
                exp=self.memory.sample()
                self.learn(exp,n_step,logger)
                
    def learn(self,experiences,n_step,logger):
        

        states,rewards,next_states,actions,dones=experiences

        full_states=states.reshape([self.batch_size,-1])
        full_nextstates=next_states.reshape([self.batch_size,-1])
        self.iter+=1
        
        for agent_number,agent in enumerate(self.multiagent):
            
            #distinguish full picture from agent picture
            state=states[:,agent_number,:]
            next_state=next_states[:,agent_number,:]            
            action=actions[:,agent_number,:]                                 
            done=dones[:,agent_number,:]
            reward=rewards[:,agent_number,:]
            
            #other agent info
            oindx=(agent_number+1)%2
            oppaction=actions[:,oindx,:]
            onstate=next_states[:,oindx,:]
            with torch.no_grad(): 
                onaction=self.multiagent[oindx].actionestimator_local(onstate)
                
            
            #calculate loss        
            Qloss=(agent.criticloss(full_states,action,oppaction,reward,next_state,onaction,next_states,done,n_step)).mean()
            
            #backwards pass
            agent.Q_optimizer.zero_grad()
            Qloss.backward()
            torch.nn.utils.clip_grad_norm_(agent.Qval_local.parameters(), 1)
            agent.Q_optimizer.step()
            

            #action loss
            actionsest=agent.actionestimator_local(state)
            intuple=(full_states,actionsest,oppaction)

            actionloss=-(agent.expectedQ(agent.Qval_local(intuple))).mean()
            agent.action_optimizer.zero_grad()
            actionloss.backward()
            agent.action_optimizer.step()
            logger.add_scalars('agent%i loss' % agent_number,{'critic loss': Qloss,
                            'actor_loss': actionloss}, self.iter)

            agent.softupdate(agent.Qval_target,agent.Qval_local)
            agent.softupdate(agent.actionestimator_target,agent.actionestimator_local)

        

    
