import torch.multiprocessing as mp
import time
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
from NNet.neuralnetwork import Policy
from UTIL.buffer import Buffer
import gym
import gc
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

GAMMA = 0.98
scaling_factor = 0.01

def actor(pipe, id, queue):
    env = gym.make('CartPole-v1')
    simulater_ = Buffer(env)
    print("start",id)
    P_network = Policy()
    count = 0
    while True:
        if pipe.poll():
            try:
                # Attempt to receive data with a timeout
                state_dict = pipe.recv()
                count = 0
                #if state_dict == 1:
                    #wait until receiving, for sync version
                #    wait = 1
                #    while wait == 1:
                #        if pipe.poll():
                #            wait = pipe.recv()
                #    continue
                if state_dict == 0:
                    break
                P_network.load_state_dict(state_dict)

            except mp.TimeoutError:
                print("No data received - non-blocking check")
                # Perform other tasks or break loop if needed
                break
            except EOFError:
                print("Pipe closed.")
                break

        #change network to bayesian version 
        #to increase entropy of state distribution which is used for regulating distributional shift
        '''
        bayesian_p = copy.deepcopy(P_network)
        with torch.no_grad():  # Ensure gradients are not computed for this operation
            for name, param in bayesian_p.named_parameters():
                # Calculate the variance of the parameter
                param_variance = torch.var(param.data)
                
                # Generate noise with variance proportional to the parameter's variance
                noise = torch.randn_like(param.data) * (param_variance.sqrt() * scaling_factor)
                
                # Add the noise to the parameter
                param.data += noise

        bayesian_p.eval()
        episode = simulater_.generate(bayesian_p)
        del bayesian_p
        '''
        if count < 30:
            episode = simulater_.generate(P_network)
            queue.put(episode)
            del episode
        count = count + 1
        #time.sleep(0.02)
        #to prevent memory issue
        time.sleep(0.1)

        torch.cuda.empty_cache()
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()
    print("actor over")
    pipe.close()

