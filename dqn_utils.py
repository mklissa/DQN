
import numpy as np

class RingBuffer(object):
    def __init__(self, max_length):
        self.max_length = max_length
        self.start = 0
        self.length = 0
        self.data = [None] * self.max_length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise KeyError('Index: {}'.format(index))
        return self.data[(self.start + index) % self.max_length]

    def append(self, value):
        if self.length < self.max_length:
            self.length += 1
        elif self.length == self.max_length:
            self.start = (self.start + 1) % self.max_length

        self.data[(self.start + self.length - 1) % self.max_length] = value
        
    def setstate(self,index, etat):
        self.data[index] = etat
      
class ExperienceMemory(object):
    def __init__(self, memory_length=10000):
        self.memory_length = memory_length
        self.actions = RingBuffer(memory_length)
        self.rewards = RingBuffer(memory_length)
        self.observations = RingBuffer(memory_length)
        self.terminals = RingBuffer(memory_length)

    def save_experience(self, observation, action, reward, done):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(done)

    def get_exp_window(self, end_index, window_size):

        observations = []

        while self.terminals[end_index - 2] == True:
            end_index = np.random.randint(window_size, len(self.observations) - 1, size=1)[0]
            
        for i in range(window_size):
            if i > 1 and self.terminals[end_index - i] == True:
                break
            observations.append(self.observations[end_index - i])

        while len(observations) < window_size:
            observations += [observations[-1]]

        observations.reverse()

        return observations,end_index

    def sample_minibatch(self, batch_size, window_size):
        full_window_size = window_size + 1
        mb_actions = []
        mb_rewards = []
        mb_first_obs = []
        mb_second_obs = []
        mb_terms = []

        last_index = len(self.observations) - 1
        window_index_ends = np.random.randint(window_size, last_index, size=batch_size)

        for end_index in window_index_ends:

            observations,end_index = self.get_exp_window(end_index, full_window_size)
            #pdb.set_trace()
            mb_first_obs += observations[0:-1]
            mb_second_obs += observations[1:]
            mb_actions.append(self.actions[end_index-1])
            mb_rewards.append(self.rewards[end_index-1])
            mb_terms.append(self.terminals[end_index-1])


        mb_first_obs = np.reshape(np.asarray(mb_first_obs), (-1, 4, 84, 84))
        mb_second_obs = np.reshape(np.asarray(mb_second_obs), (-1, 4, 84, 84))
        mb_actions = np.reshape(np.asarray(mb_actions), (-1,))
        mb_rewards = np.reshape(np.asarray(mb_rewards),(-1,))
        mb_terms = np.reshape(np.asarray(mb_terms),(-1,))

        mb_terms = np.invert(mb_terms) * 1
            
        return mb_first_obs, mb_actions, mb_rewards, mb_second_obs, mb_terms
    
