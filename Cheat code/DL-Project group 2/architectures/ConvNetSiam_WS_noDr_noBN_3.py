from torch.nn import functional as F
from torch import nn

# ********************************* Define Architecture of the model 
class ConvNetSiam_WS_noDr_noBN(nn.Module):
    # Define The Conv Network
    def __init__(self, hidden_layers):
        super(ConvNetSiam_WS_noDr_noBN, self).__init__()
        # First layer of 1 channel as input and 32 channels as output
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))


        # Second layer of 32 channel as input and 64 channels as output
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        

        # Formula to get out_put size (in_size - kernel_size + 2*(padding)) / stride) + 1
        # first layer (14-5+2*2)/1 +1 = 14/2 = 7
        # second layer (7 -4 +2*2)/1 +1 = 8/2 = 4
        # 4 * 4 * 64 input features, 1000 output features
        self.fc1 = nn.Linear(4 * 4 * 64, hidden_layers)

        # hidden_layers input features, 10 output features
        self.fc2 = nn.Linear(hidden_layers, 10)

    def forward(self, x):
        x1 = x[:, 0, :, :].view(x.size(0), 1, 14, 14)
        x2 = x[:, 1, :, :].view(x.size(0), 1, 14, 14)
        # Activation of the first convolution
        # size (batch, 32 ,7 ,7)
        x1 = self.layer1(x1)
        x2 = self.layer1(x2)

        # Activation of the first convolution
        # size (batch, 64 ,4 ,4)
        x1 = self.layer2(x1)
        x2 = self.layer2(x2)

        # Reshape (batch, 1024)
        x1 = x1.reshape(x1.size(0), -1)
        x2 = x2.reshape(x2.size(0), -1)


        # Relu activation of last layer
        x1 = F.relu(self.fc1(x1.view(-1, 4 * 4 * 64)))
        x2 = F.relu(self.fc1(x2.view(-1, 4 * 4 * 64)))

        x1 = self.fc2(x1)
        x2 = self.fc2(x2)

        return x1, x2
