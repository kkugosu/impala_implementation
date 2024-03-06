import gym
import torch
import torch.nn as nn
import torch.optim as optim
from NNet.neuralnetwork import Value, Queue, Policy
from UTIL.buffer import Buffer
from UTIL.preprocess import preprocessUnit
import gym
import copy
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

#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98
      
def main():    
    window = 10
    score = 0.0
    Trainiter = 1000
    for iter in range(Trainiter):
        cpu_policy = copy.deepcopy(P_f).cpu()
        cases = 10
        database = []
        for case in range(cases):
            episode = simulater.generate(cpu_policy)
            database = database + episode
        refine_data = preprocessUnit(database)
        state, action, reward, next_state, prob = refine_data.preprocess()
        v_value = V_f(state.type(torch.float32)).squeeze()
        v_nextvalue = V_f(next_state.type(torch.float32)).squeeze()

        pi = P_f(state.type(torch.float32))
        P = torch.gather(pi.type(torch.float32), 1, 
                    action.unsqueeze(-1)).squeeze()
        target = reward*0.01 + gamma * v_nextvalue * (reward != -1)

        delta = target - v_value
        
        lossp = -torch.log(P) * delta.squeeze().detach() 
        
        lossv = criterion(v_value.squeeze(), target.type(torch.float32).squeeze().detach())

        Poptimizer.zero_grad()
        lossp.mean().backward(retain_graph = True)
        Poptimizer.step()

        Voptimizer.zero_grad()
        lossv.mean().backward(retain_graph = True)
        Voptimizer.step()
        score = score + sum(reward)

        if iter%window==0 and iter!=0:
            print("# of episode :{}, avg score : {:.1f}".format(iter, score/(window*cases)))
            print(lossp.mean())
            print(lossv.mean())
            score = 0.0


if __name__ == '__main__':
    main()