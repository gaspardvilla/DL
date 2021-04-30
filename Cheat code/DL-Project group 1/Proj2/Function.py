import torch
from torch import Tensor, tanh
import matplotlib.pyplot as plt
import numpy as np
import math
torch.set_grad_enabled( False )

#Disclaimer: These two functions are strongly inspired of the DLC_practical file
def generate_disc_set(nb):
    input = Tensor(nb, 2).uniform_(-1, 1)
    target = 1-input.pow(2).sum(1).sub(2 / math.pi).sign().add(1).div(2).long() #1- because it was 1 outside and 0 inside
    return input, target

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp
#Disclaimer : That function is strongly inspired of the previous practical
def compute_nb_errors(model, data_input, data_target,mini_batch_size):
    nb_data_errors = 0
    for b in range(0, data_input.size(0), 1):
        output = model.forward(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


#Plot the dataset, for the outside circle, the middle point must be forgotten
def plot_Dataset(train_input,train_target):
    Label=train_target.view(-1).float() # 1- to display the other label
    x1=(train_input.narrow(1,0,1).view(-1)*Label).numpy();
    y1=(train_input.narrow(1,1,1).view(-1)*Label).numpy();
    x0=(train_input.narrow(1,0,1).view(-1)*(1-Label)).numpy();
    y0=(train_input.narrow(1,1,1).view(-1)*(1-Label)).numpy();
    plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')

    subplot=plt.subplot(1,2,1)
    axes = plt.gca();
    plt.title("Showing the dataset with the label 1");
    subplot.plot(x1,y1,'go');
    axes.set_xlim([-2,2]);
    axes.set_ylim([-2,2]);
    subplot=plt.subplot(1,2,2)
    axes = plt.gca();
    plt.title("Showing the dataset with the label 0");
    subplot.plot(x0,y0,'go');
    axes.set_xlim([-2,2]);
    axes.set_ylim([-2,2]);
    return subplot
    
#Heritage module definitiob
class Module ( object ) :
    def __init__(self):
        super().__init__()
        self.lr=0
    def forward ( self , * input ) :
        raise NotImplementedError
    def backward ( self , * gradwrtoutput ) :
        raise NotImplementedError
    def param ( self ) :
        return []

#Sequential architecture implementation
class Sequential(Module):
    def __init__(self, param , loss):
        super().__init__()
        self.model = (param)
        self.loss = loss
    def forward(self,x):
        for _object in self.model:
            x = _object.forward(x)
        return x
    
    def backward(self,y,y_pred):
        Loss=self.loss.loss(y,y_pred)
        grad_pred = self.loss.grad(y,y_pred)
        for _object in reversed(self.model):
            grad_pred = _object.backward(grad_pred)
        return Loss
    def print(self):
        for _object in (self.model):
            _object.print()
    def set_Lr(self,lr=0):
        for _object in self.model:
            if isinstance(_object,Linear):
                _object.set_Lr(lr)
                
#Linear layer implementation
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.x = torch.zeros(out_features)        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.rand(size=(self.in_features,self.out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias= torch.rand(self.out_features)
        self.bias.data.uniform_(-stdv, stdv)

    def print(self):
            print([self.weight.size(0),self.weight.size(1)])
            print(self.x.size(0))
    def print_weight(self):
        print(self.weight) 
    def update(self, grad):
        lr=self.lr
        self.weight = self.weight - lr * self.x.t().mm(grad) 
        self.bias = self.bias - lr * grad * 1
    
    def backward(self,grad):
        b = grad.mm(self.weight.t())
        self.update(grad)
        return b
    def forward(self,x):
        self.x = x
        return x.mm(self.weight)+self.bias;
    def set_Lr(self, lr):
        self.lr=lr
        return
#Relu activation function implementation
class ReLu(Module):
    
    def __init__(self ):
        super().__init__()
        self.save=0;
    def forward(self,x):
        y = x.clamp(min = 0)
        self.save=x;
        return y
    def backward(self,x):
        y=self.save>0
        return y.float()*x
         
    def print(self):
        return
#tanh activation function implementation

class Tanh(Module):
    def __init__(self, ):
        super().__init__()        
    def forward(self,x):
        y = (x.exp() - (-x).exp()) / ( x.exp() +  (-x).exp())
        return y
    def backward(self,x):
        y=4 * (x.exp() + x.mul(-1).exp()).pow(-2)
        return y
    def print(self):
        return
        
#MSE Loss implementation 
class LossMSE(Module):
    def __init__(self, ):
        super().__init__()
        
    def loss(self,y,y_pred):
        loss = (y_pred - y).pow(2).sum()
        return loss
    
    def grad(self,y,y_pred):
        return 2*(y_pred-y)