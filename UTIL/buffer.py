import gym
import numpy as np
import torch

env = gym.make('CartPole-v1')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class Buffer:
    def __init__(self, env):

        self.env = env

    def generate(self, policy):
        #policy = _policy.cpu()
        state = env.reset()[0]
        total_rewards = 0
        episode = []
        step = 0
        while step < 500:

            action_p = policy(torch.from_numpy(state)) # action = distribution
            action_p = action_p.detach()
            action = torch.multinomial(action_p, num_samples=1).item()
            next_state, reward, done, _ , _ = env.step(action)
            if done == True:
                reward = -1
                episode.append((state, action, reward, next_state, action_p))
                break
            episode.append((state, action, reward, next_state, action_p))
            state = next_state
            total_rewards += reward
            
            step = step + 1

        return episode
    

