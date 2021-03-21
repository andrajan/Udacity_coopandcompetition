import time
from collections import deque
import numpy as np
import torch
from multiagent import MultiAgent
from torch.utils.tensorboard import SummaryWriter



def train(env,param):
    start=time.time()
    scores = []
    avgscores=[]
    scores_window = deque(maxlen=100)  # last 100 scores
    logger=SummaryWriter() #initialize tensorboard logger
    
    #First we unpack our parameters
    printyn=param['print']
    n_steps=param['n_step']
    n_episodes=param['n_episodes']
    noise=param['noise']
    param=param['agentparam']
    param['print']=printyn
    agent = MultiAgent(param)
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state=env_info.vector_observations
        score = 0
        culreward=0
        steps=0
        
        while True:
            action = agent.act(state)
            action = [act.detach().numpy() for act in action]
            env_info=env.step(action)[brain_name]
            next_state = env_info.vector_observations
            reward = np.array(env_info.rewards)                   
            done = np.array(env_info.local_done) 
            #culmative reward with discount gamma applied
            culreward+=reward.reshape(-1,1)*param['gamma']**steps
            steps+=1
            
            if steps==n_steps or np.any(done):
                agent.step(state, action, culreward, next_state, done,n_steps,logger)
                steps=0
                culreward=0


            state = next_state
            score += np.max(reward)

            if np.any(done):
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score) 
        avgscores.append(np.mean(scores_window))
        logger.add_scalar('score',score,i_episode)
        if printyn:
            print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_window), score), end="")
        if i_episode % 100 == 0:
            for ag_number,ag in enumerate(agent.multiagent):
                torch.save(ag.actionestimator_local.state_dict(), 'checkpoint_actor%i.pth' % ag_number )
                torch.save(ag.Qval_local.state_dict(), 'checkpoint_critic%i.pth' % ag_number)
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))  
        if np.mean(scores_window)>3 and i_episode>100:
            ttime=time.time()-start
            for ag_number,ag in enumerate(agent.multiagent):
                torch.save(ag.actionestimator_local.state_dict(), 'checkpoint_actor%i.pth' % ag_number )
                torch.save(ag.Qval_local.state_dict(), 'checkpoint_critic%i.pth' % ag_number)

            print('Agent took {} hours and {} minutes to solve enviroment in {} episodes'.format(
                int(ttime/3600),int(ttime%60),i_episode))
            return scores
    logger.close()
    return scores,avgscores

