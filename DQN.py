import sys
import pdb
import time
import argparse

from dqn_utils import RingBuffer, ExperienceMemory
from theano_utils import ConvLayer, HiddenLayer

import theano
from theano import tensor as T
import keras

import gym
import numpy as np
import random
import cv2
from six.moves import cPickle

from collections import deque, namedtuple, OrderedDict

from PIL import Image as imm


class DQN():

    def __init__(self, env_name, epsilon_start, maximum, crop):
        self.epsilon = epsilon_start
        self.frame_width = 84  
        self.frame_height = 84 
        self.window_size = 4
        self.env = gym.make(env_name)
        self.n_actions = self.env.action_space.n
        self.maximum= maximum
        self.mode='train'

        self.crop = crop
        
        self.gamename = env_name[:-16].lower()
        
        if self.gamename == 'spaceinvaders' or self.gamename == 'boxing':
            self.maximum=True
        if self.gamename == 'spaceinvaders':
            self.crop = False
            
        if self.maximum:
            self.env._step = self._step_max
        else:
            self.env._step = self._step
        
        clip_val = 1.
        self.phis = T.tensor4()
        self.mask = T.matrix()
        self.targets = T.matrix()
        self.phis_prime = T.tensor4()
        
        self.q_net,self.qparams = self.get_network(self.n_actions,self.phis)
        self.t_net,self.tparams = self.get_network(self.n_actions,self.phis_prime)
        
        error = self.targets - self.q_net.output
        clip_condition = T.abs_(error) < clip_val
        
        squared_loss = .5 * T.square(error)
        linear_loss = clip_val * (T.abs_(error) - .5 * clip_val)
        
        switch_loss  = T.switch(clip_condition,squared_loss,linear_loss)
        mask_loss = T.sum(switch_loss * self.mask,axis=-1)
        loss = T.sum(mask_loss)
        
#        updates = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01).get_updates(params=self.qparams,
#                                                                         loss=loss,
#                                                                         constraints={})

        
        updates = self.create_rms_updates(self.qparams,loss)
        
        self.predik = theano.function([self.phis],self.q_net.output)
        self.predik_t = theano.function([self.phis_prime],self.t_net.output)
            
        self.trainit = theano.function([self.phis,self.targets,self.mask],
                                       loss,
                                       updates=updates)
    
    def create_rms_updates(self,params,loss):
        
        grads = T.grad(loss,params)
        
        updates = OrderedDict()
        rho=theano.shared(np.asarray(0.95, dtype='float32'))
        epsilon=theano.shared(np.asarray(0.01, dtype='float32'))
        lr = theano.shared(np.asarray(0.00025, dtype='float32'))
    
        for param, grad in zip(params, grads):
    
            
            acc = theano.shared(np.zeros(param.get_value(borrow=True).shape,
                                                         dtype=theano.config.floatX),
                                       broadcastable=param.broadcastable)
            acc_up = rho * acc + (1.-rho) * T.square(grad)
            updates[acc] = acc_up
    
            updates[param] =  param - lr * grad /(T.sqrt(acc_up)  + epsilon) 
        
        
        return updates
    
    def get_network(self,n_actions,input_obs):
        
        conv0 = ConvLayer( input=input_obs,
                                image_shape=(None, 4, 84, 84),
                                filter_shape=(32, 4, 8, 8),
                                layer_num=0,
                                stride=(4,4))
        
        conv1 = ConvLayer( input=conv0.output,
                                    image_shape=(None, None, None, None),
                                filter_shape=(64, 32, 4, 4),
                                layer_num=1,
                                stride=(2,2))
        
        conv2 = ConvLayer( input=conv1.output,
                                    image_shape=(None, None, None, None),
                                filter_shape=(64, 64, 3, 3),
                                layer_num=2,
                                stride=(1,1))
        
        fc0 = HiddenLayer(input=conv2.output.flatten(2),
                                   n_in=3136,
                                 n_out=512,
                                 layer_name='fc0')
        
    
        lastlayer = HiddenLayer(input=fc0.output,
                                         n_in=512,
                                 n_out=n_actions,
                                 layer_name='last_layer',
                                activation=None)
        
        params = conv0.params + conv1.params+ conv2.params + fc0.params + lastlayer.params
        
        return lastlayer, params
    

    def _step(self,a):
        reward = 0.0
        action = self.env.unwrapped._action_set[a]
        lives_before = self.env.unwrapped.ale.lives()
        for _ in range(4):
            reward += self.env.unwrapped.ale.act(action)
        ob = self.env.unwrapped._get_obs()
        done = self.env.unwrapped.ale.game_over() or (self.mode == 'train' and lives_before != self.env.unwrapped.ale.lives())
        return ob, reward, done, {}

    def _step_max(self,a):
        reward = 0.0
        action = self.env.unwrapped._action_set[a]
        lives_before = self.env.unwrapped.ale.lives()
        for i in range(4):
            if i == 3:
                ob = self.env.unwrapped._get_obs()
            reward += self.env.unwrapped.ale.act(action)
    
        ob = np.maximum(self.env.unwrapped._get_obs(),ob)
        
        done = self.env.unwrapped.ale.game_over() or  (self.mode == 'train' and lives_before != self.env.unwrapped.ale.lives())
        return ob, reward, done, {}

        
    def reshape_observation(self,rgb_img):
  
        if self.crop:
            img = cv2.resize(rgb_img, (84, 110),interpolation=cv2.INTER_LINEAR)
            unused_height = 110 - 84
            bottom_crop = 8
            top_crop = unused_height - bottom_crop
            img = img[top_crop: 110 - bottom_crop, :]
        else:
            img = cv2.resize(rgb_img, (84, 84),interpolation=cv2.INTER_LINEAR)

        #Luminance
        img = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.0722 + img[:, :, 2] * 0.7152
    

        return np.array(img).astype('uint8')


    def get_action(self,recent_observations):
        
        while len(recent_observations) < self.window_size:
            recent_observations.insert(0, recent_observations[0])
            
        processed_observations = np.reshape(recent_observations, 
                                            (-1, self.window_size, 
                                             self.frame_width, 
                                             self.frame_height)).astype('float32') / 255.


        q_values = self.predik(processed_observations).squeeze()
        
        action = np.random.randint(0, self.n_actions) if np.random.uniform() < self.epsilon else np.argmax(q_values)
        return action

    def clip_reward(self,reward):
        return np.clip(reward, -1., 1.)






    def deep_q_loop(self,epsilon_start,epsilon_end,epsilon_steps,batch_size):
    

        np.random.seed(np.random.randint(0,2**32))
        experience_replay =ExperienceMemory(memory_length=1000000)

        ########################################################################
        print("Time to warm up...")
        
        state = self.env.reset()

        
        recent_observations = deque(maxlen=self.window_size)
        recent_observations.append(self.reshape_observation(state))
        
        done=False


        for i in range(50000):

            sys.stdout.write("Populating memory... " + str(i)+ "\r")
                
            action = self.get_action(recent_observations)
            next_state,reward,done,_ = self.env.step(action)
            
            experience_replay.save_experience(recent_observations[-1],action,reward,done)
            recent_observations.append(self.reshape_observation(next_state))

            if done:
                state = self.env.reset()
                recent_observations = deque(maxlen=self.window_size)
                recent_observations.append(self.reshape_observation(state))
        ########################################################################
        
        print("\nTraining time...")
        
        total_t= 0
        running_reward = None
        
        for ep in range(200):
            self.mode='train'
            epoch_t = 0
            rewards=0
            state = self.env.reset()
            recent_observations = deque(maxlen=self.window_size)
            recent_observations.append(self.reshape_observation(state))

            while epoch_t < 250000:
                total_t += 1
                epoch_t += 1
                
                if total_t % 10000 == 0:
                    print("weights are upated...")
                    for qp,tp in zip(self.qparams,self.tparams):
                        tp.set_value(qp.get_value())

    
                if self.epsilon > epsilon_end: self.epsilon -= ( epsilon_start - epsilon_end )/epsilon_steps
    
                action = self.get_action(recent_observations)
                next_state, reward, done, _ = self.env.step(action)
                
                experience_replay.save_experience(recent_observations[-1], action, reward, done)
                recent_observations.append(self.reshape_observation(next_state))
                rewards+=reward
            
                if total_t % 4 ==0:
                    targets = np.zeros((batch_size, self.n_actions))
                    masks = np.zeros((batch_size, self.n_actions))
    
                    batch_states, batch_actions, batch_rewards, \
                    batch_next_states, batch_terminals = experience_replay.sample_minibatch(batch_size, self.window_size)
                    
                    
                    reshaped_states = batch_states.astype('float32') * (1./255.)
                    reshaped_next_states = batch_next_states.astype('float32') * (1./255.)
                    clipped_rewards = self.clip_reward(batch_rewards)
    
                    target_net_values = self.predik_t(reshaped_next_states)
                    qnet_values = self.predik(reshaped_next_states)
                    q_choices = np.argmax(qnet_values,axis=1)
                    target_net_values = target_net_values[range(batch_size),q_choices].flatten()
                    
                    

                    
                    discounted_reward_batch = 0.99 * target_net_values
                    discounted_reward_batch *= batch_terminals
                    Rs = batch_rewards + discounted_reward_batch
                    
                    for (target, mask, R, action) in zip(targets, masks, Rs, batch_actions):
                        target[action] = R 
                        mask[action] = 1.  
                    
                    targets = np.array(targets).astype('float32')
                    masks = np.array(masks).astype('float32')    
                    
                    
                    self.trainit(reshaped_states,targets,masks)
                    
        
                if done:
                    running_reward = rewards if running_reward is None else running_reward * 0.99 + rewards * 0.01
                    print('Epoch: {} Total steps: {}, reward: {} running_mean: {} epsilon: {}'.format(
                        ep, total_t, rewards, running_reward, self.epsilon))
                    rewards=0
                    state = self.env.reset()
                    recent_observations = deque(maxlen=self.window_size)
                    recent_observations.append(self.reshape_observation(state))


            filename =self.gamename + '_epoch' + str(ep) + '.pkl'
            with open(filename,'wb') as f:
                cPickle.dump(self.qparams,f)
            
            #########################################################
            print("Testing after end of epoch:")

            state = self.env.reset()
            recent_observations = deque(maxlen=self.window_size)
            recent_observations.append(self.reshape_observation(state))
            
            self.mode = 'test'
            training_eps = self.epsilon
            self.epsilon = 0.05 # test time epsilon
            rewards=0
            tot_rewards=0
            num_episodes=0
            train_steps=0
            done=False


            while train_steps < 12500:
                train_steps +=1
                action = self.get_action(recent_observations)
                next_state,reward,done,_ = self.env.step(action)
                recent_observations.append(self.reshape_observation(next_state))
                rewards += reward
                tot_rewards+= reward
                
                if done or train_steps >= 12500:
                    num_episodes+=1
                    print("Reward of {} for episode {} at testing time.".format(rewards,num_episodes))
                    rewards=0
                    state = self.env.reset()
                    recent_observations = deque(maxlen=self.window_size)
                    recent_observations.append(self.reshape_observation(state))
            
            
            print("Epoch {} with average reward per episode of {} at test time.".format(ep,tot_rewards/num_episodes))
            self.epsilon = training_eps # Bring back training mode epsilon

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='BreakoutDeterministic-v3')
    parser.add_argument('--epsilon_start', type=float, default='1.')
    parser.add_argument('--epsilon_end', type=float, default='0.1')
    parser.add_argument('--epsilon_steps', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--maximum', type=bool, default=False)
    parser.add_argument('--crop', type=bool, default=True)
    

    args = parser.parse_args()
    
    agent = DQN(args.env,
                    args.epsilon_start,
                    args.maximum,
                    args.crop)

    agent.deep_q_loop(args.epsilon_start,
                      args.epsilon_end,
                      args.epsilon_steps,
                      args.batch_size)

if __name__ == '__main__':
    main()
