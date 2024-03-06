import gym
import torch
import torch.nn as nn
import torch.optim as optim
from NNet.neuralnetwork import Value, Queue, Policy
from UTIL.buffer import Buffer
from UTIL.preprocess import preprocessUnit
import gym
import copy
import warnings
import time
import numpy as np
import gc
from torch.utils.tensorboard import SummaryWriter


gamma         = 0.98
warnings.filterwarnings("ignore", category=DeprecationWarning, message="`np.bool8` is a deprecated alias")
env = gym.make('CartPole-v1')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

V_f = Value().to(device)
Q_f = Queue().to(device)
P_f = Policy().to(device)
simulater = Buffer(env)

Voptimizer = optim.Adam(V_f.parameters(), lr=0.0002)
Qoptimizer = optim.Adam(Q_f.parameters(), lr=0.0002)
Poptimizer = optim.Adam(P_f.parameters(), lr=0.0002)
criterion = nn.MSELoss(reduction='sum')
kl_loss = nn.KLDivLoss(reduction="batchmean")
      
def trainer(con1, con2, con3, con4, queue):   
    writer = SummaryWriter() 
    
    window = 10
    score = 0.0
    total_case = 0
    Trainiter = 10000
    total_dblen = 0
    prev_mean = torch.tensor([0, 0,  0,  0], device='cuda:0',dtype=torch.float64)
    prev_var = torch.tensor([0, 0,  0,  0], device='cuda:0',dtype=torch.float64)
    for iter in range(Trainiter):
        cpu_policy = copy.deepcopy(P_f).eval()
        state_dict = cpu_policy.state_dict()
        con1.send(state_dict)
        con2.send(state_dict)
        con3.send(state_dict)
        con4.send(state_dict)
        del cpu_policy
        torch.cuda.empty_cache()
        with torch.no_grad():
            torch.cuda.empty_cache()
        gc.collect()
        #to prevent fluctuation of batch size
        time.sleep(0.02)
        #print("sended")
        cases = 0
        datastart = time.time()
        database = []
        '''
        for case in range(cases):
            episode = simulater.generate(cpu_policy)
            database = database + episode
        '''
        while True:
            try:
                # Attempt to get an item from the queue
                # Use a relatively short timeout value
                episode = queue.get(timeout=0.0)
                database = database + episode
                cases = cases + 1
            except :
                # If no item is retrieved within the timeout
                # assume the queue is empty (or no items are currently available)
                if len(database) != 0:
                    break
        
        #for sync version
        #con1.send(1)
        #con2.send(1)
        #con3.send(1)
        #con4.send(1)
        if len(database) > 3000:
            refine_data = preprocessUnit(database[:3000])
            total_dblen = total_dblen + 3000
        else:
            refine_data = preprocessUnit(database)
            total_dblen = total_dblen + len(database)

        state, action, reward, next_state, prob = refine_data.preprocess()
        dataend = time.time()

        forwardstart = time.time()
        v_value = V_f(state.type(torch.float32)).squeeze()
        v_nextvalue = V_f(next_state.type(torch.float32)).squeeze()
        pi = P_f(state.type(torch.float32))
        P = torch.gather(pi.type(torch.float32), 1, 
                    action.unsqueeze(-1)).squeeze()
        P_past = torch.gather(prob.type(torch.float32), 1, 
                    action.unsqueeze(-1)).squeeze()
        IS = (P/P_past).detach()
        # target is Q function
        target = reward*0.01 + gamma * v_nextvalue * (reward != -1)
        # Q - V, V is unbiased
        v_value = v_value*IS
        target = target*IS
        delta = target - v_value
        lossp = -torch.log(P) * delta.squeeze().detach() 
        lossv = criterion(v_value.squeeze(), target.type(torch.float32).squeeze().detach())
        
        entropy = torch.sum(-torch.log(pi) * pi, -1).mean()

        forwardend = time.time()
        backwardstart = time.time()

        Voptimizer.zero_grad()
        lossv.mean().backward(retain_graph = True)
        Voptimizer.step()
        
        kld_policy_loss = kl_loss(torch.log(pi), prob.detach())


        """
        #nomal version
        Poptimizer.zero_grad()
        lossp.mean().backward()
        Poptimizer.step()
        """


        """
        #trpo version
        policyloss = lossp.mean() + kld_policy_loss
        Poptimizer.zero_grad()
        policyloss.backward(retain_graph=True)
        Poptimizer.step()
        """

        #"""
        #sac version
        Poptimizer.zero_grad()
        policyloss = lossp.mean() - entropy*0.01
        policyloss.backward(retain_graph = True)
        Poptimizer.step()
        #"""
        
        # to calculate state kld
        mean = torch.mean(state, dim=0)
        var = torch.var(state, dim=0)
        kl_divergence = torch.log(var.sqrt() / prev_var.sqrt()) + (prev_var + (prev_mean - mean).pow(2)) / (2 * var) - 0.5
        
        # Sum or average the KL divergences of all dimensions if you want a single measure
        state_kl_divergence = torch.sum(kl_divergence)

        prev_mean = mean
        prev_var = var
        del mean, var, kl_divergence
        backwardend = time.time()
        score = score + sum(reward)
        total_case = total_case + cases

        if iter%window==0 and iter!=0:
            print("# of episode :{}, avg score : {:.1f}".format(iter, score/(total_case)))
            print(lossp.mean())
            print(lossv.mean())
            print("dblen", total_dblen)
            print("entropy",entropy)
            print("kldpolicyloss", kld_policy_loss)
            print(torch.cuda.memory_allocated(device))
            score = 0.0
            total_case = 0
            total_dblen = 0
        writer.add_scalar('Loss/policy', lossp.mean(), iter)
        writer.add_scalar('Loss/value', lossv.mean(), iter)
        writer.add_scalar('Data/database_size', len(database), iter)
        writer.add_scalar('IS/min', torch.min(IS), iter)
        writer.add_scalar('IS/avg', torch.mean(IS), iter)
        writer.add_scalar('IS/max', torch.max(IS), iter)
        writer.add_scalar('performance', sum(reward)/cases, iter)
        writer.add_scalar('time/learnerbackward', backwardend - backwardstart, iter)
        writer.add_scalar('time/learnerforward', forwardend - forwardstart, iter)
        writer.add_scalar('time/learnerbatch', dataend - datastart, iter)
        writer.add_scalar('entropy', entropy, iter)
        writer.add_scalar('kld/policy', kld_policy_loss, iter)
        writer.add_scalar('kld/state', state_kl_divergence, iter)
        writer.add_scalar('memory', torch.cuda.memory_allocated(device), iter)
        #for sync version
        #con1.send(2)
        #con2.send(2)
        #con3.send(2)
        #con4.send(2)
        torch.save(P_f.state_dict(), "./PARAM/policy.pth")
        torch.save(V_f.state_dict(), "./PARAM/value.pth")


