import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. /np.sqrt(fan_in)
    return (-lim, lim)
    
class Critic(nn.Module):
    """Critic (Value) Model."""
  
    def __init__(self, state_size, action_size, hidden_units):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes of hidden layers
        """
        super().__init__()
        units = hidden_units.copy()
        self.input = nn.Linear(state_size, units[0])
        self.batch_norm = nn.BatchNorm1d(units[0])
        units[0] += action_size
        self.hidden_layers = nn.ModuleList([nn.Linear(size_in, size_out) for size_in, size_out in zip(units[:-1], units[1:])])
        self.output = nn.Linear(units[-1], 1)
        
    def reset_parameter(self):
        self.input.weight.data.uniform(*hidden_init(self.input))
        for layer in self.hidden_layers:
            layer.weight.data.uniform(*hidden_init(layer))
        self.output.weight.data.uniform(-3e-3, 3e-3)
        
    def forward(self, state, action):
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x0 = F.relu(self.input(state))
        x0 = self.batch_norm(x0)
        x = torch.cat((x0, action), dim=1)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output(x)