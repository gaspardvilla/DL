#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import dlc_practical_prologue as prologue

from torch import optim
from torch.nn import functional as F
from torch import nn


# In[ ]:


def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        _, _, result = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(result, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


# In[44]:


def train_model(model, train_input, train_target, train_classes, nb_epochs, mini_batch_size):
    criterion = nn.CrossEntropyLoss()
    eta = 1e-2
    optimizer = optim.SGD(model.parameters(), lr = eta)
    
    for e in range(nb_epochs):    
        
        for b in range(0, train_input.size(0), mini_batch_size):
            digit1, digit2, result = model(train_input.narrow(0, b, mini_batch_size))
            
            loss_result = criterion(result, train_target.narrow(0, b, mini_batch_size))
            loss_digit1 = criterion(digit1, train_classes[:,0].narrow(0, b, mini_batch_size))
            loss_digit2 = criterion(digit2, train_classes[:,1].narrow(0, b, mini_batch_size))
            loss = loss_result + 10*loss_digit1 + 10*loss_digit2
            
            model.zero_grad()
            loss.backward()
            optimizer.step()


# In[48]:


class ConvNoWS(nn.Module):
    def __init__(self):
        super(ConvNoWS, self).__init__()
        
        #Input channels = 1, output channels = 32
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2))
        
        #Input channels = 32, output channels = 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2))
        
        # Formula to get out_put size (in_size - kernel_size + 2*(padding)) / stride) + 1
        # first layer (14 - 3 + 2*1) + 1 = 14/2 = 7
        # second layer (7 - 2 + 2*1) + 1 = 8/2 = 4
        # 4 * 4 * 64 input features, 1000 output features
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 64, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10))

        #Comparison of the two digits
        self.layer_comp = nn.Sequential(
            nn.Linear(20, 200),
            nn.ReLU(),
            nn.Linear(200, 200),
            nn.ReLU(),
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


# In[ ]:


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


# In[54]:


def main_Conv_WS_AL(nb_epochs):
    
    model = ConvNoWS()
    mini_batch_size = 100
    Train_error = []
    Test_error = []
    
    train_input, train_target, train_classes,_, _, _     = prologue.generate_pair_sets(1000)

    for i in range(1, nb_epochs+1):
    
        train_model(model, train_input, train_target, train_classes, i, mini_batch_size)
        L = get_tests(10)

        nb_train_errors = compute_nb_errors(model, train_input, train_target, mini_batch_size)
        Train_error.append(nb_train_errors/10)

        nb_moy_test_error = 0

        for k in range (0, len(L)):
            nb_test_errors = compute_nb_errors(model, L[k][0], L[k][1], mini_batch_size)
            nb_moy_test_error += nb_test_errors

        nb_moy_test_error = nb_moy_test_error/len(L)
        Test_error.append(nb_moy_test_error/10)
        
    return Train_error, Test_error


# In[55]:


Train_error, Test_error = main_Conv_WS_AL(10)
print(Train_error)
print(Test_error)


# In[ ]:




