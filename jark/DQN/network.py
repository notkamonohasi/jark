from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

class DQN_Network(nn.Module) : 
    def __init__(self, state_dimension, action_dimension, target_mode, target_learning_rate) : 
        super().__init__()
        self.target_mode = target_mode
        self.target_learning_rate = target_learning_rate
        self.fc1 = nn.Linear(state_dimension, 24)
        self.fc2 = torch.nn.Linear(24, 24)
        self.fc3 = torch.nn.Linear(24, 24)
        self.fc4 = torch.nn.Linear(24, action_dimension)

    def forward(self, x : tensor) -> tensor : 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def __call__(self, x : tensor) -> tensor : 
        return self.forward(x)  

    def update_target(self, source : DQN_Network) : 
        if self.target_mode == False : 
            print("unexpected ActorNetwork::update_target ERROR!!")
            return 
        else : 
            for target_param, param in zip(self.parameters(), source.parameters()) :
                target_param.data.copy_(target_param.data * (1.0 - self.target_learning_rate) + param.data * self.target_learning_rate)

    def inititalize_target(self, source : DQN_Network) : 
        if self.target_mode == False : 
            print("unexpected initialize_target ERROR!!")
            return 
        else : 
            for target_param, param in zip(self.parameters(), source.parameters()) :
                target_param.data.copy_(param.data)