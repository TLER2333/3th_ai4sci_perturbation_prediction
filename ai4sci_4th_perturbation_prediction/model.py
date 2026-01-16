# simple MLP model
import torch
import torch.nn as nn
import torch.nn.functional as F

class PerturbationModel(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim=512):
        super(PerturbationModel, self).__init__()
       
        self.dropout = nn.Dropout(0.2) 
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        identity = x 
        
        out = F.relu(self.fc1(x))
        out = self.dropout(out) 
        
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return identity + out