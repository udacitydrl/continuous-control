import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. /np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_size, hidden_units):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes of hidden layers
        """
        super().__init__()
        self.input = nn.Linear(state_size, hidden_units[0])
        self.batch_norm = nn.BatchNorm1d(hidden_units[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(size_in, size_out) for size_in, size_out in zip(hidden_units[:-1], hidden_units[1:])])
        self.output = nn.Linear(hidden_units[-1], action_size)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.input.weight.data.uniform_(*hidden_init(self.input))
        for layer in self.hidden_layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, state):
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = F.relu(self.input(state))
        x = self.batch_norm(x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return F.tanh(self.output(x))
    
        
        
                                            
        
    
