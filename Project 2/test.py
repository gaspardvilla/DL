import torch
import math

from Supplementary.Modules import *
from Supplementary.Functions import *

# Just use for the plots at the end
import matplotlib.pyplot as plt
from numpy import exp,arange
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

torch.set_grad_enabled(False)

# Initialization of our principal parameters
nb_epochs = 100
mini_batch_size = 10

# This gives the nb of models that we will train and test to have conclusive
# results.
nb_rounds = 2

# Initialization of training set
train_input, train_classes, _, _ = create_problem(1000)

# initialization of our test sets
Tests = get_tests(nb_rounds)

# initialization of all the models
Model_ReLU_MSE, Model_Tanh_MSE, Model_Sigmoid_MSE, Model_Leaky_ReLU_MSE, Model_ELU_MSE, \
        Model_Tanh_Dropout_MSE, Model_Tanh_Dropout_CE = \
                get_Models(LossMSE(), nb_rounds)

# -------------------------------------------------------------------------- #

# Activation function as ReLU, with MSE loss
Train_error_ReLU_MSE, Test_error_ReLU_MSE, std_deviation_ReLU_MSE = \
    train_and_test_model(Model_ReLU_MSE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with ReLU and LossMSE 
print("Average train_error on", nb_rounds,"Sequential models with ReLU and LossMSE is",Train_error_ReLU_MSE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with ReLU and LossMSE 
print("Average test_error on", nb_rounds,"Sequential models with ReLU and LossMSE is",Test_error_ReLU_MSE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_ReLU_MSE, "%")

# -------------------------------------------------------------------------- #

# Activation function as Tanh, with MSE loss
Train_error_Tanh_MSE, Test_error_Tanh_MSE, std_deviation_Tanh_MSE = \
    train_and_test_model(Model_Tanh_MSE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with Tanh and LossMSE
print("Average train_error on", nb_rounds,"Sequential models with Tanh and LossMSE is",Train_error_Tanh_MSE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with Tanh and LossMSE 
print("Average test_error on", nb_rounds,"Sequential models with Tanh and LossMSE is",Test_error_Tanh_MSE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_Tanh_MSE, "%")

# -------------------------------------------------------------------------- #

# Activation function as a Sigmoid, with MSE loss
Train_error_Sigmoid_MSE, Test_error_Sigmoid_MSE, std_deviation_Sigmoid_MSE = \
    train_and_test_model(Model_Sigmoid_MSE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with Sigmoid and LossMSE 
print("Average train_error on", nb_rounds,"Sequential models with Sigmoid and LossMSE is",Train_error_Sigmoid_MSE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with Sigmoid and LossMSE 
print("Average test_error on", nb_rounds,"Sequential models with Sigmoid and LossMSE is",Test_error_Sigmoid_MSE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_Sigmoid_MSE, "%")

# -------------------------------------------------------------------------- #

# Activation function as a Leaky ReLU, with MSE loss
Train_error_Leaky_ReLU_MSE, Test_error_Leaky_ReLU_MSE, std_deviation_Leaky_ReLU_MSE = \
    train_and_test_model(Model_Leaky_ReLU_MSE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with Leaky_ReLU and LossMSE 
print("Average train_error on", nb_rounds,"Sequential models with Leaky_ReLU and LossMSE is",Train_error_Leaky_ReLU_MSE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with Leaky_ReLU and LossMSE 
print("Average test_error on", nb_rounds,"Sequential models with Leaky_ReLU and LossMSE is",Test_error_Leaky_ReLU_MSE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_Leaky_ReLU_MSE, "%")

# -------------------------------------------------------------------------- #

# Activation function as ELU, with MSE loss
Train_error_ELU_MSE, Test_error_ELU_MSE, std_deviation_ELU_MSE = \
    train_and_test_model(Model_ELU_MSE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with ELU and LossMSE 
print("Average train_error on", nb_rounds,"Sequential models with ELU and LossMSE is",Train_error_ELU_MSE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with ELU and LossMSE 
print("Average test_error on", nb_rounds,"Sequential models with ELU and LossMSE is",Test_error_ELU_MSE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_ELU_MSE, "%")

# -------------------------------------------------------------------------- #

# Activation function as Tanh, with Cross Entropy (CE) loss
Train_error_Tanh_CE, Test_error_Tanh_CE, std_deviation_Tanh_CE = \
    train_and_test_model(Model_Tanh_CE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with Tanh and Cross-Entropy Loss
print("Average train_error on", nb_rounds,"Sequential models with Tanh and Cross-Entropy Loss is",Train_error_Tanh_CE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with Tanh and Cross-Entropy Loss
print("Average test_error on", nb_rounds,"Sequential models with Tanh and Cross-Entropy Loss is",Test_error_Tanh_CE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_Tanh_CE, "%")

# -------------------------------------------------------------------------- #
# -------------------------------------------------------------------------- #
# Activation function as Sigmoid, with Cross Entropy (CE) loss
Train_error_Sigmoid_CE, Test_error_Sigmoid_CE, std_deviation_Sigmoid_CE = \
    train_and_test_model(Model_Sigmoid_CE, train_input, train_classes, Tests, nb_epochs, mini_batch_size)

#Average train_error (%) on number_of_rounds Sequential models with Sigmoid and Cross-Entropy Loss
print("Average train_error on", nb_rounds,"Sequential models with Sigmoid and Cross-Entropy Loss is",Train_error_Sigmoid_CE[0],'%')

#Average test_error (%) on number_of_rounds Sequential models with Sigmoid and Cross-Entropy Loss
print("Average test_error on", nb_rounds,"Sequential models with Sigmoid and Cross-Entropy Loss is",Test_error_Sigmoid_CE[0],"%")

#Standard deviation corresponding to the average on test_error just above
print("Standard deviation corresponding to the average on test_error just above is ", std_deviation_Sigmoid_CE, "%")

# -------------------------------------------------------------------------- #