import torch.multiprocessing as mp
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from NNet.neuralnetwork import Queue, Policy
from simulater import actor

from async_ac import trainer
import gym
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, message="`np.bool8` is a deprecated alias")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    # Create pipes and queue
    p_con1, c_con1 = mp.Pipe()
    p_con2, c_con2 = mp.Pipe()
    p_con3, c_con3 = mp.Pipe()
    p_con4, c_con4 = mp.Pipe()
    queue = mp.Queue(maxsize=1000)
    # Start the actor process

    trainer_process = mp.Process(target = trainer, args=(p_con1, p_con2, p_con3, p_con4, queue))
    trainer_process.daemon = True
    trainer_process.start()

    time.sleep(1)

    actor_process1 = mp.Process(target=actor, args=(c_con1, 1, queue))
    actor_process1.daemon = True
    actor_process1.start()

    actor_process2 = mp.Process(target=actor, args=(c_con2, 2, queue))
    actor_process2.daemon = True
    actor_process2.start()

    actor_process3 = mp.Process(target=actor, args=(c_con3, 3, queue))
    actor_process3.daemon = True
    actor_process3.start()

    actor_process4 = mp.Process(target=actor, args=(c_con4, 4, queue))
    actor_process4.daemon = True
    actor_process4.start()


    actor_process1.join()
    actor_process2.join()
    actor_process3.join()
    actor_process4.join()
    trainer_process.join()


