# NN initializer
import torch.multiprocessing as mp
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from UTIL.preprocess import preprocessUnit
from NNet.neuralnetwork import Policy, Queue
from UTIL.buffer import Buffer
import gym
import copy

def train(con1, con2, con3, con4, queue):
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.set_device(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    GAMMA = 0.98
    P_network = Policy().to(device)
    Q_network = Queue().to(device)
    #P_network.load_state_dict(torch.load("./PARAM/policy.pth"))
    #Q_network.load_state_dict(torch.load("./PARAM/queue.pth"))
    kl_loss = nn.KLDivLoss(reduction="batchmean")

    env = gym.make('CartPole-v1')

    Poptimizer = optim.SGD(P_network.parameters(), lr=0.0001, weight_decay=0.000001)
    Qoptimizer = optim.SGD(Q_network.parameters(), lr=0.001, weight_decay=0.000001)
    criterion = nn.MSELoss(reduction='mean')

    whole_train = 0
    total_dbtime = 0
    total_traintime = 0

    sendnetwork = copy.deepcopy(P_network).eval()
    state_dict = sendnetwork.state_dict()
    con1.send(state_dict)
    con2.send(state_dict)
    con3.send(state_dict)
    con4.send(state_dict)

    while whole_train < 10000:
        time.sleep(0.2)
        #sendnetwork = copy.deepcopy(P_network).eval()
        #state_dict = sendnetwork.state_dict()
        
        
        # build database
        # state, action, reward, next_state, probability
        start_time = time.time()
        database = []
    
        cases = 0
        
        """
        for case in range(cases):
            episode = simulater.generate(P_network)
            database = database + episode
        """
        while True:
            try:
                # Attempt to get an item from the queue
                # Use a relatively short timeout value
                item = queue.get(timeout=0.0)
                database = database + item
                cases = cases + 1
            except :
                # If no item is retrieved within the timeout
                # assume the queue is empty (or no items are currently available)
                if len(database) != 0:
                    break
        
        end_time = time.time()
        dbtime = end_time - start_time
        con1.send(1)
        con2.send(1)
        con3.send(1)
        con4.send(1)

        start_time = time.time()
        refine_data = preprocessUnit(database)
        state, action, reward, next_state, prob = refine_data.preprocess()

        prob2 = P_network(state.type(torch.float32).detach())
        #close = torch.isclose(prob, prob2, atol=0.0001)
        #print("aa", torch.where(~close))
        important_s = prob2/prob
        important_s = torch.gather(important_s, 1, action.unsqueeze(-1)).detach().squeeze()
        #total_traintime = total_traintime + train_time

        next_action = torch.multinomial(P_network(next_state.type(torch.float32)), 1).squeeze()
        
        P = torch.gather(P_network(state.type(torch.float32)), 1, 
                        action.unsqueeze(-1)).squeeze()

        BaseQ = torch.gather(Q_network(state.type(torch.float32)), 1, 
                            action.unsqueeze(-1)).squeeze()
        
        NextQ = torch.gather(Q_network(next_state.type(torch.float32)), 1, 
                            next_action.unsqueeze(-1)).squeeze().clone().detach()
        
        Ploss = -torch.sum(torch.log(P) * BaseQ.clone().detach())/len(database)

        TargetQvalue = NextQ* (reward != -1)*GAMMA + (reward*0.1).float()
        
        BaseQ = BaseQ * important_s
        TargetQvalue = TargetQvalue * important_s

        Qloss = criterion(BaseQ, TargetQvalue)

        Qoptimizer.zero_grad()
        Qloss.backward(retain_graph=True)
        for param in Q_network.parameters():
                    param.grad.data.clamp_(-1, 1)
        Qoptimizer.step()

        #if Qloss < 0.001:
        Poptimizer.zero_grad()
        Ploss.backward(retain_graph=True)
        for param in P_network.parameters():
                    param.grad.data.clamp_(-1, 1)
        Poptimizer.step()
        
        prob2 = P_network(state.type(torch.float32).detach())
        # kld term like trpo
        kl_pg_loss = kl_loss(torch.log(prob2), prob.detach())*1e+2
        #print("pgloss", kl_pg_loss)
        Poptimizer.zero_grad()
        kl_pg_loss.backward(retain_graph=True)
        for param in P_network.parameters():
            param.grad.data.clamp_(-1, 1)
        Poptimizer.step()

        end_time = time.time()

        train_time = end_time - start_time
        
        torch.save(P_network.state_dict(), "./PARAM/policy.pth")
        torch.save(Q_network.state_dict(), "./PARAM/queue.pth")

        if whole_train%10 == 0:
            print("dblen == ",len(database))
            print(Ploss)
            print(Qloss)
            print("kl", kl_pg_loss)
            print(len(database)/cases)
            print("dbtime", dbtime)
            print("traintime", train_time)
        
        sendnetwork = copy.deepcopy(P_network).eval()
        state_dict = sendnetwork.state_dict()
        con1.send(2)
        con2.send(2)
        con3.send(2)
        con4.send(2)
        time.sleep(0.1)
        con1.send(state_dict)
        con2.send(state_dict)
        con3.send(state_dict)
        con4.send(state_dict)

        whole_train = whole_train + 1
        
    env.close()
    # Close the connection and wait for the receiver to finish
    con1.send(0)
    con2.send(0)
    con3.send(0)
    con4.send(0)
    time.sleep(1)
    print("trainer over")
    con1.close()
    con2.close()
    con3.close()
    con4.close()

