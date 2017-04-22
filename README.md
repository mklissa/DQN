# DQN


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

The pickle files must be named after the game (i.e. breakout.pkl). You can also change that directly in the script.
As mentionned in this [blog](https://blog.openai.com/adversarial-example-research/) from OpenAI, DQN, as well as more recent algorithms, are subject to adversarial attacks. To see the effects on this implementation, choose:

`python play.py --env [environment name] --adv 1`

# Results


<p align="center" size="width 150">
  <img src="https://github.com/mklissa/DQN/blob/master/results/pong.gif" width="150"/>

  <img src="https://github.com/mklissa/DQN/blob/master/results/breakout.gif" width="150"/>
</p>
<p align="center" size="width 150">
  <img src="https://github.com/mklissa/DQN/blob/master/results/space.gif" width="150"/>

  <img src="https://github.com/mklissa/DQN/blob/master/results/boxing.gif" width="150"/>
</p>
