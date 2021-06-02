# Import of troch library
import torch
from torch import nn

# Definition of an architecture of neural networks with convolution network, 
# weight sharing and auxiliary loss
class Conv_WS_AL_D(nn.Module):
    def __init__(self):
        
        # Set if there is or not auxiliary loss (AL) for this architecture
        self.AL = True
        
        super(Conv_WS_AL_D, self).__init__()
        
        p = 0.1 # Dropout rate
        
        #Input channels = 1, output channels = 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.MaxPool2d(kernel_size=2))
        
        #Input channels = 32, output channels = 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.MaxPool2d(kernel_size=2))
        
        # Formula to get out_put size (in_size - kernel_size + 2*(padding)) / stride) + 1
        # first layer (14 - 3 + 2*1) + 1 = 14/2 = 7
        # second layer (7 - 2 + 2*1) + 1 = 8/2 = 4
        # 4 * 4 * 64 input features, 1000 output features
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 64, 1000),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(1000, 10))

        #Comparison of the two digits
        self.layer_comp = nn.Sequential(
            nn.Linear(20, 200),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(200, 2))
        
    def forward(self, x):
        
        first_digit = x[:,[0]]
        second_digit = x[:,[1]]

        first_digit = self.layer1(first_digit)
        second_digit = self.layer1(second_digit)
        
        first_digit = self.layer2(first_digit)
        second_digit = self.layer2(second_digit)
    
        first_digit = self.fc(first_digit.view(-1, 4 * 4 * 64))
        second_digit = self.fc(second_digit.view(-1, 4 * 4 * 64))
        
        result = torch.cat((first_digit, second_digit), dim=1, out=None)
        result = self.layer_comp(result)
        
        return first_digit, second_digit, result