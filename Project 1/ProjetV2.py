# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 09:11:04 2021

@author: aurel
"""
import torch
import dlc_practical_prologue as prologue

from torch import optim
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable


######################################################################
def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


######################################################################
def train_model(model, train_input, train_target, nb_epochs, mini_batch_size):
    criterion = nn.CrossEntropyLoss()
    eta = 1e-3
    optimizer = optim.SGD(model.parameters(), lr = eta)

    for e in range(nb_epochs):
        acc_loss = 0
        
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            acc_loss = acc_loss + loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
    
        print(e, acc_loss)
        
            
######################################################################            
# def eval_Model(model, mini_batch_size, nb_epochs):
    




######################################################################   
class FirstConvNet(nn.Module):
    def __init__(self):
        super(FirstConvNet, self).__init__()
        #Input channels = 2, output channels = 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        #Input channels = 32, output channels = 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        
        # Formula to get out_put size (in_size - kernel_size + 2*(padding)) / stride) + 1
        # first layer (14-5+2*2)/1 +1 = 14/2 = 7
        # second layer (7 -4 +2*2)/1 +1 = 8/2 = 4
        # 4 * 4 * 64 input features, 1000 output features
        self.fc1 = nn.Linear(4 * 4 * 64, 1000)
        
        # 1000 input features, 2 output features
        self.fc2 = nn.Linear(1000, 2)
        
    def forward(self, x):
        # Activation of the first convolution 
        # size (batch, 32 ,7 ,7)
        out = self.layer1(x)
        
        # Activation of the first convolution 
        # size (batch, 64 ,4 ,4)
        out = self.layer2(out)
        
        # Reshape (batch, 1024)
        out = out.reshape(out.size(0), -1)
        
        
        # Relu activation of last layer 
        out = F.relu(self.fc1(out.view(-1,4 * 4 * 64)))
        
        out = self.fc2(out)
        return out
    
######################################################################   
    
train_input, train_target, train_classes, test_input, test_target, test_classes \
    = prologue.generate_pair_sets(1000)
    
# train_input, train_target, train_classes \
#     = Variable(train_input), Variable(train_target), Variable(train_classes)
# test_input, test_target, test_classes \
#     = Variable(test_input), Variable(test_target), Variable(test_classes)

def get_tests(n):
    M = []
    for k in range (0, n):
        L = []
        _, _, _, test_input, test_target, test_classes =  prologue.generate_pair_sets(1000)
        L.append(test_input)
        L.append(test_target)
        L.append(test_classes)
        M.append(L)
    return M


model = FirstConvNet()
nb_epochs = 25
mini_batch_size = 100

train_model(model, train_input, train_target, nb_epochs, mini_batch_size)
L = get_tests(10)
average_nb_test_error = 0
for k in range (0, len(L)):
    nb_test_errors = compute_nb_errors(model, L[k][0], L[k][1], mini_batch_size)
    nb_moy_test_error += nb_test_errors
    print('test error FirstConvNet {:0.2f}% {:d}/{:d}'.format((100 * nb_test_errors) / L[k][0].size(0),
                                                nb_test_errors, L[k][0].size(0)))
print('Average test error FirstConvNet {:0.2f}% {:0.1f}/{:d}'.format((100*nb_moy_test_error/10) / L[0][0].size(0),
                                                                  nb_moy_test_error/10, L[0][0].size(0) ))

