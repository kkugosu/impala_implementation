import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class preprocessUnit:
    def __init__(self, dataset):
        self.database = dataset

            
    def extract_state(self):
        statelist = []
        for i in range(len(self.database)):
            statelist = np.append(statelist, self.database[i][0])
        statelist = np.reshape(statelist, (-1, 4))
        return statelist

    def extract_action(self):
        actionlist = []
        for i in range(len(self.database)):
            actionlist = np.append(actionlist, self.database[i][1])
        return actionlist

    def extract_reward(self):
        actionlist = []
        for i in range(len(self.database)):
            actionlist = np.append(actionlist, self.database[i][2])
        return actionlist

    def extract_next_state(self):
        statelist = []
        for i in range(len(self.database)):
            statelist = np.append(statelist, self.database[i][3])
        statelist = np.reshape(statelist, (-1, 4))
        return statelist

    def extract_probability(self):
        statelist = torch.tensor([])
        for i in range(len(self.database)):
            statelist = torch.cat((statelist, self.database[i][-1] ))
        statelist = statelist.view(-1, 2)
        return statelist
    
    def preprocess(self):
        state = torch.from_numpy(self.extract_state()).to(device)
        action = torch.tensor(self.extract_action(), dtype=torch.int64).to(device)
        reward = torch.tensor(self.extract_reward()).to(device)
        next_state = torch.from_numpy(self.extract_next_state()).to(device)
        prob = self.extract_probability().to(device) #used for importance sampling and entropy
        
        return state, action, reward, next_state, prob
                
        