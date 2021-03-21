[//]: # (Image References)



#  Udacity Collaboration and Competition

### Introduction

This project solves thes tennis environment. In the Tennis environment there are two agents; one for each of the rackets. Every time the racket connects with the ball and send the ball over the net it gets a reward of +0.1. If it lets the ball hit the ground or hits the ball out of bounds however, it gets a reward of -0.01. The total score is counted as the maximum reward accumulated over an episode of the two agents. If over 100 episodes the model achieves a score of over 0.5 then the enviroment is considered solved. In our code it will train till it achieves a score of 3, which we can achieve.

Each agent has its own observation space that records the velocity and position of the ball and racket, and can respond with two continuous actions; adjusting its position from the net and jumping.

To solve this enviroment we implement MADDPG based on a paper by [*Lowe et al* ](https://arxiv.org/pdf/1706.02275.pdf). We also include the ability to implement the same multi agent structure but with the D4PG algorithm based on a paper by [*Barth-Maron et al*](https://arxiv.org/pdf/1804.08617.pdf). However for us the D4PG did not work as well.
### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Make sure the correct dependencies are installed, and this app is in your local folder. Follow the isntructions to create a python environment with the correct dependencies in the following link [drlnd](https://github.com/udacity/deep-reinforcement-learning#dependencies). If you want to run tensorboard, install this as well.

### Implementation

Head over to Report.ipynb to eather train, view our results via tensorboard or watch our trained agent. 
