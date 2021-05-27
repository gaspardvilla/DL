import torch

from torch import optim
from torch.nn import functional as F
from torch import nn

# Definition of an architecture of neural networks with MLP, weight sharing
# and auxiliary loss

class MLP_WS_AL_BD(nn.Module):
    def __init__(self):
        
        # Set if there is or not auxiliary loss (AL) for this architecture
        self.AL = True
        
        super(MLP_WS_AL_BD, self).__init__()
        
        nb_hidden = 100
        input_size = 14*14
        p = 0.1 # Dropout rate
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, nb_hidden),
            nn.BatchNorm1d(nb_hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(nb_hidden, nb_hidden),
            nn.BatchNorm1d(nb_hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(nb_hidden, nb_hidden),
            nn.BatchNorm1d(nb_hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(nb_hidden, 10),
            nn.LogSoftmax(dim=1)
        )
        
        self.layers_comp = nn.Sequential(
            nn.Linear(20, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(200, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(200, 2)
        )
        
    def forward(self, x):
        first_digit = x[:,[0]]
        second_digit = x[:,[1]]
        
        first_digit = first_digit.view(first_digit.size(0),-1) #torch.reshape() can also be used
        second_digit = second_digit.view(second_digit.size(0),-1)
        
        first_digit = self.layers(first_digit)
        second_digit = self.layers(second_digit)
        
        result = torch.cat((first_digit, second_digit), dim=1, out=None)
        result = self.layers_comp(result)
    
        return first_digit, second_digit, result