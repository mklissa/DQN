# DQN
Implementation of DQN and DDQN in Theano

This is an implementation of the classic DQN algorithm using Theano. 

# Requirements

- Python 3.6
- Theano 0.8.2
- OpenCV
- OpenAI Gym
- Numpy

Trained weights are provided for the following games: Boxing, Breakout, Pong and SpaceInvaders. 

To train the model, please use:

`python DQN.py --env [environment name]`

To see it play:

`python play.py --env [environment name]`

Notice that the pickle files must be named after the game (i.e. breakout.pkl). You can also change that directly in the script.

# Results


<p align="center" size="width 150">
  <img src="https://github.com/mklissa/DQN/blob/master/results/pong.gif" width="150"/>
</p>
<p align="center" size="width 150">
  <img src="https://github.com/mklissa/DQN/blob/master/results/breakout.gif" width="150"/>
</p>
<p align="center" size="width 150">
  <img src="https://github.com/mklissa/DQN/blob/master/results/space.gif" width="150"/>
</p>
<p align="center" size="width 150">
  <img src="https://github.com/mklissa/DQN/blob/master/results/boxing.gif" width="150"/>
</p>
