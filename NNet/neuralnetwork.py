import torch
from torch import nn
import torch.nn.functional as F

class Queue(nn.Module):
    def __init__(self):
        super(Queue, self).__init__()
        self.fc1 = nn.Linear(4, 256)    
        self.fc3 = nn.Linear(256, 2)     
        self.llu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.llu(self.fc1(x))
        x = self.fc3(x)
        return x
    
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 256)    
        self.fc3 = nn.Linear(256, 2)     
        self.llu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.llu(self.fc1(x))
        x = self.softmax(self.fc3(x))
        return x
    
class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(4, 256)    
        self.fc3 = nn.Linear(256, 1)     
        self.llu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.llu(self.fc1(x))
        x = self.fc3(x)
        return x
    

