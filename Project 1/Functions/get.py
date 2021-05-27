# Import the different architectures
# MLP
from Architectures.MLP_NoWS_NoAL import MLP_NoWS_NoAL
from Architectures.MLP_NoWS_AL import MLP_NoWS_AL
from Architectures.MLP_WS_NoAL import MLP_WS_NoAL
from Architectures.MLP_WS_AL import MLP_WS_AL
from Architectures.MLP_WS_AL_B import MLP_WS_AL_B
from Architectures.MLP_WS_AL_D import MLP_WS_AL_D
from Architectures.MLP_WS_AL_BD import MLP_WS_AL_BD
# Conv
from Architectures.Conv_NoWS_NoAL import Conv_NoWS_NoAL
from Architectures.Conv_NoWS_AL import Conv_NoWS_AL
from Architectures.Conv_WS_NoAL import Conv_WS_NoAL
from Architectures.Conv_WS_AL import Conv_WS_AL
from Architectures.Conv_WS_AL_B import Conv_WS_AL_B
from Architectures.Conv_WS_AL_D import Conv_WS_AL_D
from Architectures.Conv_WS_AL_BD import Conv_WS_AL_BD


# This file is for having a list of n models that share the same architecture. 
# It is usefull for us for the estimates that we want to have in our report,
# because, it is required that our estimates mus be done on more than 10 rounds.
# And, clearly the rounds must be random, so we use these functions.


# Function for getting a list of n models with Conv_NoWS_NoAL architecture
def conv_nows_noal(n):
    L = []
    for k in range (0, n):
        L.append(Conv_NoWS_NoAL())
    return L


# Function for getting a list of n models with Conv_NoWS_AL architecture
def conv_nows_al(n):
    L = []
    for k in range (0, n):
        L.append(Conv_NoWS_AL()) 
    return L


# Function for getting a list of n models with Conv_WS_NoAL architecture
def conv_ws_noal(n):
    L = []
    for k in range (0, n):
        L.append(Conv_WS_NoAL())
    return L


# Function for getting a list of n models with Conv_WS_AL architecture
def conv_ws_al(n):
    L = []
    for k in range (0, n):
        L.append(Conv_WS_AL())
    return L

# Function for getting a list of n models with Conv_WS_AL_B architecture
def conv_ws_al_b(n):
    L = []
    for k in range (0, n):
        L.append(Conv_WS_AL_B())
    return L


# Function for getting a list of n models with Conv_WS_AL_D architecture
def conv_ws_al_d(n):
    L = []
    for k in range (0, n):
        L.append(Conv_WS_AL_D())
    return L


# Function for getting a list of n models with Conv_WS_AL_BD architecture
def conv_ws_al_bd(n):
    L = []
    for k in range (0, n):
        L.append(Conv_WS_AL_BD())
    return L


# Function for getting a list of n models with MLP_NoWS_NoAL architecture
def mlp_nows_noal(n):
    L = []
    for k in range (0, n):
        L.append(MLP_NoWS_NoAL())
    return L


# Function for getting a list of n models with MLP_NoWS_AL architecture
def mlp_nows_al(n):
    L = []
    for k in range (0, n):
        L.append(MLP_NoWS_AL())
    return L


# Function for getting a list of n models with MLP_WS_NoAL architecture
def mlp_ws_noal(n):
    L = []
    for k in range (0, n):
        L.append(MLP_WS_NoAL())
    return L


# Function for getting a list of n models with MLP_WS_AL architecture
def mlp_ws_al(n):
    L = []
    for k in range (0, n):
        L.append(MLP_WS_AL())
    return L


# Function for getting a list of n models with MLP_WS_AL_B architecture
def mlp_ws_al_b(n):
    L = []
    for k in range (0, n):
        L.append(MLP_WS_AL_B())
    return L


# Function for getting a list of n models with MLP_WS_AL_D architecture
def mlp_ws_al_d(n):    
    L = [] 
    for k in range (0, n):
        L.append(MLP_WS_AL_D())
    return L


# Function for getting a list of n models with MLP_WS_AL_BD architecture
def mlp_ws_al_bd(n):
    L = []
    for k in range (0, n):
        L.append(MLP_WS_AL_BD())
    return L