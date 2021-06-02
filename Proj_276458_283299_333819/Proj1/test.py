import torch
import dlc_practical_prologue as prologue

# Import of functions 
from Functions.get_tests import get_tests
from Functions import get
from Functions.train_and_test_model import train_and_test_model
from Functions.digit_normalization import digit_normalization

# Just used here for the final plots in the report
import matplotlib.pyplot as plt


# Initialization of all the parameters
nb_epochs = 50
mini_batch_size = 100

# Initialization of all the parameters
# !! IMPORTANT !! To run the test.py file much faster, we recommend using a 
# number of epochs 25 and a number of rounds = 2
nb_rounds = 20

# Definition of the train set
train_input, train_target, train_classes,_, _, _ \
    = prologue.generate_pair_sets(1000)

# Normalization of the train set
train_input = digit_normalization(train_input)

# Definition of the nb_rounds tests sets
Tests = get_tests(nb_rounds)


# -------------------------------------------------------------------------- #


# This part is for the plot to compare different architectures for MLP

print('MLP_NoWS_NoAL')
# Test error for NoWS_NoAL
NoWS_NoAL = get.mlp_nows_noal(nb_rounds)
_, Test_error_NoWS_NoAL, std_deviation_nows_noal \
    = train_and_test_model(NoWS_NoAL, train_input, train_target, 
                           train_classes, Tests, nb_epochs, mini_batch_size)

print('MLP_NoWS_AL')
# Test error for NoWS_AL
NoWS_AL = get.mlp_nows_al(nb_rounds)
_, Test_error_NoWS_AL, std_deviation_nows_al \
    = train_and_test_model(NoWS_AL, train_input, train_target, 
                           train_classes, Tests, nb_epochs, mini_batch_size)

print('MLP_WS_NoAL')
# Test error for WS_NoAL
WS_NoAL = get.mlp_ws_noal(nb_rounds)
_, Test_error_WS_NoAL, std_deviation_ws_noal \
    = train_and_test_model(WS_NoAL, train_input, train_target, 
                           train_classes, Tests, nb_epochs, mini_batch_size)

print('MLP_WS_AL')
# Test error for WS_AL
WS_AL = get.mlp_ws_al(nb_rounds)
_, Test_error_WS_AL, std_deviation_ws_al \
    = train_and_test_model(WS_AL, train_input, train_target, 
                           train_classes, Tests, nb_epochs, mini_batch_size)

# Standard deviation of the tests error
print("For NoWS_NoAL, after", nb_epochs, "epochs, there is", Test_error_NoWS_NoAL[-1], 
      "% of test error and a standard deviation equal to", std_deviation_nows_noal)
print("For NoWS_AL, after", nb_epochs, "epochs, there is", Test_error_NoWS_AL[-1], 
      "% of test error and a standard deviation equal to", std_deviation_nows_al)
print("For WS_NoAL, after", nb_epochs, "epochs, there is", Test_error_WS_NoAL[-1], 
      "% of test error and a standard deviation equal to", std_deviation_ws_noal)
print("For WS_AL, after", nb_epochs, "epochs, there is", Test_error_WS_AL[-1], 
      "% of test error and a standard deviation equal to", std_deviation_ws_al)
    
# Plots
plt.figure(1)
epochs = torch.linspace(1, nb_epochs, steps=nb_epochs)
plt.plot(epochs, Test_error_NoWS_NoAL, label='Test error for MLP_NoWS_NoAL')
plt.plot(epochs, Test_error_NoWS_AL, label='Test error for MLP_NoWS_AL')
plt.plot(epochs, Test_error_WS_NoAL, label='Test error for MLP_WS_NoAL')
plt.plot(epochs, Test_error_WS_AL, label='Test error for MLP_WS_AL')
plt.xlabel('Epochs')
plt.ylabel('Percentage of error [%]')
plt.title('Test error for different architectures with MLP')
plt.legend()

plt.savefig('MLP_vs_architectures.jpg')


# -------------------------------------------------------------------------- #


# This part is for the plot to compare different architectures for Conv

print('Conv_NoWS_NoAL')
# Test error for NoWS_NoAL
NoWS_NoAL = get.conv_nows_noal(nb_rounds)
_, Test_error_NoWS_NoAL, std_deviation_nows_noal \
    = train_and_test_model(NoWS_NoAL, train_input, train_target, 
                           train_classes, Tests, nb_epochs, mini_batch_size)

print('Conv_NoWS_AL')
# Test error for NoWS_AL
NoWS_AL = get.conv_nows_al(nb_rounds)
_, Test_error_NoWS_AL, std_deviation_nows_al \
    = train_and_test_model(NoWS_AL, train_input, train_target, 
                           train_classes, Tests, nb_epochs, mini_batch_size)

print('Conv_WS_NoAL')
# Test error for WS_NoAL
WS_NoAL = get.conv_ws_noal(nb_rounds)
_, Test_error_WS_NoAL, std_deviation_ws_noal \
    = train_and_test_model(WS_NoAL, train_input, train_target, 
                           train_classes, Tests, nb_epochs, mini_batch_size)

print('Conv_WS_AL')
# Test error for WS_AL
WS_AL = get.conv_ws_al(nb_rounds)
_, Test_error_WS_AL, std_deviation_ws_al \
    = train_and_test_model(WS_AL, train_input, train_target, 
                           train_classes, Tests, nb_epochs, mini_batch_size)

# Standard deviation of the tests error
print("For NoWS_NoAL, after", nb_epochs, "epochs, there is", Test_error_NoWS_NoAL[-1], 
      "% of test error and a standard deviation equal to", std_deviation_nows_noal)
print("For NoWS_AL, after", nb_epochs, "epochs, there is", Test_error_NoWS_AL[-1], 
      "% of test error and a standard deviation equal to", std_deviation_nows_al)
print("For WS_NoAL, after", nb_epochs, "epochs, there is", Test_error_WS_NoAL[-1], 
      "% of test error and a standard deviation equal to", std_deviation_ws_noal)
print("For WS_AL, after", nb_epochs, "epochs, there is", Test_error_WS_AL[-1], 
      "% of test error and a standard deviation equal to", std_deviation_ws_al)
    
# Plots
plt.figure(2)
epochs = torch.linspace(1, nb_epochs, steps=nb_epochs)
plt.plot(epochs, Test_error_NoWS_NoAL, label='Test error for Conv_NoWS_NoAL')
plt.plot(epochs, Test_error_NoWS_AL, label='Test error for Conv_NoWS_AL')
plt.plot(epochs, Test_error_WS_NoAL, label='Test error for Conv_WS_NoAL')
plt.plot(epochs, Test_error_WS_AL, label='Test error for Conv_WS_AL')
plt.xlabel('Epochs')
plt.ylabel('Percentage of error [%]')
plt.title('Test error for different architectures with Conv')
plt.legend()

plt.savefig('Conv_vs_architectures.jpg')


# -------------------------------------------------------------------------- #


# This part is for the plot with MLP_NoWS_NoAL and Conv_NoWS_NoAL architectures

print('Conv_B')
# Train and test error for MLP
B = get.conv_ws_al_b(nb_rounds)
_, Test_error_B, std_deviation_b \
    = train_and_test_model(B, train_input, train_target, 
                            train_classes, Tests, nb_epochs, mini_batch_size)

print('Conv_D')
# Train and test error for MLP
D = get.conv_ws_al_d(nb_rounds)
_, Test_error_D, std_deviation_d \
    = train_and_test_model(D, train_input, train_target, 
                            train_classes, Tests, nb_epochs, mini_batch_size)

print('Conv_BD')
# Train and test error for MLP
BD = get.conv_ws_al_bd(nb_rounds)
_, Test_error_BD, std_deviation_bd \
    = train_and_test_model(BD, train_input, train_target, 
                            train_classes, Tests, nb_epochs, mini_batch_size)

print('Conv_Noth')
# Train and test error for MLP
Noth = get.conv_ws_al(nb_rounds)
_, Test_error_Noth, std_deviation_noth \
    = train_and_test_model(Noth, train_input, train_target, 
                            train_classes, Tests, nb_epochs, mini_batch_size)

# Standard deviation of the tests error
print("For Noth, after", nb_epochs, "epochs, there is", Test_error_Noth[-1], 
      "% of test error and a standard deviation equal to", std_deviation_noth)
print("For B, after", nb_epochs, "epochs, there is", Test_error_B[-1], 
      "% of test error and a standard deviation equal to", std_deviation_b)
print("For D, after", nb_epochs, "epochs, there is", Test_error_D[-1], 
      "% of test error and a standard deviation equal to", std_deviation_d)
print("For BD, after", nb_epochs, "epochs, there is", Test_error_BD[-1], 
      "% of test error and a standard deviation equal to", std_deviation_bd)

# Plots
plt.figure(3)
epochs = torch.linspace(1, nb_epochs, steps=nb_epochs)
plt.plot(epochs, Test_error_Noth, label='Test error for Conv_WS_AL')
plt.plot(epochs, Test_error_B, label='Test error for Conv_WS_AL_B')
plt.plot(epochs, Test_error_D, label='Test error for Conv_WS_AL_D')
plt.plot(epochs, Test_error_BD, label='Test error for Conv_WS_AL_BD')
plt.xlabel('Epochs')
plt.ylabel('Percentage of error [%]')
plt.title('Test error for different architectures with Conv')
plt.legend()

plt.savefig('Conv_B_D_BD.jpg')


# -------------------------------------------------------------------------- #


# This part is for the plot with MLP_NoWS_NoAL and Conv_NoWS_NoAL architectures

print('MLP_B')
# Train and test error for MLP
B = get.mlp_ws_al_b(nb_rounds)
_, Test_error_B, std_deviation_b \
    = train_and_test_model(B, train_input, train_target, 
                            train_classes, Tests, nb_epochs, mini_batch_size)

print('MLP_D')
# Train and test error for MLP
D = get.mlp_ws_al_d(nb_rounds)
_, Test_error_D, std_deviation_d \
    = train_and_test_model(D, train_input, train_target, 
                            train_classes, Tests, nb_epochs, mini_batch_size)

print('MLP_BD')
# Train and test error for MLP
BD = get.mlp_ws_al_bd(nb_rounds)
_, Test_error_BD, std_deviation_bd \
    = train_and_test_model(BD, train_input, train_target, 
                            train_classes, Tests, nb_epochs, mini_batch_size)

print('MLP_Noth')
# Train and test error for MLP
Noth = get.mlp_ws_al(nb_rounds)
_, Test_error_Noth, std_deviation_noth \
    = train_and_test_model(Noth, train_input, train_target, 
                            train_classes, Tests, nb_epochs, mini_batch_size)

# Standard deviation of the tests error
print("For Noth, after", nb_epochs, "epochs, there is", Test_error_Noth[-1], 
      "% of test error and a standard deviation equal to", std_deviation_noth)
print("For B, after", nb_epochs, "epochs, there is", Test_error_B[-1], 
      "% of test error and a standard deviation equal to", std_deviation_b)
print("For D, after", nb_epochs, "epochs, there is", Test_error_D[-1], 
      "% of test error and a standard deviation equal to", std_deviation_d)
print("For BD, after", nb_epochs, "epochs, there is", Test_error_BD[-1], 
      "% of test error and a standard deviation equal to", std_deviation_bd)

# Plots
plt.figure(4)
epochs = torch.linspace(1, nb_epochs, steps=nb_epochs)
plt.plot(epochs, Test_error_Noth, label='Test error for MLP_WS_AL')
plt.plot(epochs, Test_error_B, label='Test error for MLP_WS_AL_B')
plt.plot(epochs, Test_error_D, label='Test error for MLP_WS_AL_D')
plt.plot(epochs, Test_error_BD, label='Test error for MLP_WS_AL_BD')
plt.xlabel('Epochs')
plt.ylabel('Percentage of error [%]')
plt.title('Test error for different architectures with MLP')
plt.legend()

plt.savefig('MLP_B_D_BD.jpg')


# -------------------------------------------------------------------------- #


# This part is for the plot to compare the best architecture for MLP and Conv

print('MLP')
# Train and test error for MLP
MLP = get.mlp_ws_al_bd(nb_rounds)
Train_error_MLP, Test_error_MLP, std_deviation_mlp \
    = train_and_test_model(MLP, train_input, train_target, 
                            train_classes, Tests, nb_epochs, mini_batch_size)

print('Conv')
# Train and test error for Conv
Conv = get.conv_ws_al_b(nb_rounds)
Train_error_Conv, Test_error_Conv, std_deviation_conv \
    = train_and_test_model(Conv, train_input, train_target, 
                            train_classes, Tests, nb_epochs, mini_batch_size)

# Standard deviation of the tests error
print("For MLP, after", nb_epochs, "epochs, there is", Test_error_MLP[-1], 
      "% of test error and a standard deviation equal to", std_deviation_mlp)
print("For Conv, after", nb_epochs, "epochs, there is", Test_error_Conv[-1], 
      "% of test error and a standard deviation equal to", std_deviation_conv)
    
# Plots
plt.figure(5)
epochs = torch.linspace(1, nb_epochs, steps=nb_epochs)
plt.plot(epochs, Train_error_MLP, 'C3--', label='Train error for MLP_WS_AL_BD')
plt.plot(epochs, Test_error_MLP, 'C3', label='Test error for MLP_WS_AL_BD')
plt.plot(epochs, Train_error_Conv, 'C0--', label='Train error for Conv_WS_AL_B')
plt.plot(epochs, Test_error_Conv, 'C0', label='Test error for Conv_WS_AL_B')
plt.xlabel('Epochs')
plt.ylabel('Percentage of error [%]')
plt.title('Train and test error for different architectures')
plt.legend()

plt.savefig('MLP_WS_AL_BD_vs_Conv_WS_AL_B.jpg')


# -------------------------------------------------------------------------- #