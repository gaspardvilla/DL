import torch
import math

from Supplementary.Functions import *
from Supplementary.Modules import *

# Just use for the plots at the end
import matplotlib.pyplot as plt
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

torch.set_grad_enabled(False)


train_input, train_classes, _, _ = create_problem(1000)
#print(train_input.size())
#print(train_input.narrow(0, b, mini_batch_size).size())
nb_epochs = 100
mini_batch_size = 10

model = Sequential([Linear(2,25), Leaky_ReLU(), Linear(25,25), Leaky_ReLU(), Linear(25,1), Leaky_ReLU()], LossMSE())
model.lr_method("Adam", 1.0e-3)
train_model(model, train_input, train_classes, nb_epochs, mini_batch_size)



nb_train_errors = compute_nb_errors(model, train_input, train_classes, mini_batch_size)
print('train error {:0.2f}% {:f}/{:f}'.format((100 * nb_train_errors) / train_input.size(0), nb_train_errors, train_classes.size(0)))

L = get_tests(10)
average_nb_test_error = 0
for k in range (0, len(L)):
    nb_test_errors = compute_nb_errors(model, L[k][0], L[k][1], mini_batch_size)
    average_nb_test_error += nb_test_errors
    print('test error {:0.2f}% {:f}/{:f}'.format((100 * nb_test_errors) / L[k][0].size(0), nb_test_errors, L[k][0].size(0)))
print('Average test error {:0.2f}% {:0.1f}/{:d}'.format((100*average_nb_test_error/len(L)) / L[0][0].size(0), average_nb_test_error/len(L), L[0][0].size(0)))