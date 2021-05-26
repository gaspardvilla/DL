import torch
import math

torch.set_grad_enabled(False)

# -------------------------------------------------------------------------- #

# This function return a 2D tensor that is the rando selection of inputs for our
# stochastic gradient method, taking in count the number of mini_batches.

# We suppose here that our mini_batch_size is well chosen taking in count the fact
# that it divides input_size.
def create_random_batch(input_size, mini_batch_size):
    
    # Initialization
    N = int(input_size / mini_batch_size)
    new_batch = torch.ones(N, mini_batch_size)
    indices = torch.randperm(input_size)
    
    # Cut the tensor in N parts
    for k in range(N):
        new_batch[k] = indices[k * mini_batch_size : (k+1) * mini_batch_size]
    
    return new_batch

# -------------------------------------------------------------------------- #

# This function will train the model on nb_epochs epochs with specified
# train_input, train_classes and a mini_batch_size.
def train_model(model, train_input, train_classes, nb_epochs, mini_batch_size):
    
    for epoch in range(nb_epochs):
        # Get a N x n tensor of indices that make N "list" of n random indices
        # for the input tensor. We note n the mini_batch_size and N the number
        # of inputs divided by n. 
        random_batches = create_random_batch(train_input.size(0), mini_batch_size).long()
        
        for batch in random_batches:
            # Get the output of a sample of train_input given by the model
            output = model.forward(train_input[batch])
            # Train the model with this output tqking in count the real classes
            loss = model.backward(output, train_classes[batch])

# -------------------------------------------------------------------------- #

# This function compute the number of error the model made with data_input as
# input and data_target as target taking in count if we use the cross entropy
# loss of the MSE loss.
def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        result = model.forward(data_input.narrow(0, b, mini_batch_size))
        
        if model.loss.is_MSE():
            # If the loss function is MSE
            predicted_classes = (result >= 0.5).int()
        else:
            # If the loss function is CrossEntropy
            _, predicted_classes = torch.max(result, 1)
        
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
                
    return nb_data_errors

# -------------------------------------------------------------------------- #

# This function gives train and test sets for the giving problem of the project
# that is to classifies points in [0,1]^2 if there are in a circle with a
# radius equal to R = 1 / sqrt(2pi)
def create_problem(nb_samples):
    
    # Remark: the function .uniform return a uniform distribution on [0,1) instead of [0,1],
    # but in our case it's not a problem since it is only a train and a test set on a circle
    # that do not touch the border of the set [0,1]^2.
    train_input = torch.empty(nb_samples, 2).uniform_(0, 1)
    test_input = torch.empty(nb_samples, 2).uniform_(0, 1)
    
    # Radius of our circle
    R = 1 / math.sqrt(2 * math.pi)
    
    # In this order: substract 0.5 (centered in (0,0)), take the square,
    # sum it and substract R^2 (distance of the point from the center of the 
    # circle). Now, we get the sign to know if it's outside or inside the 
    # circle and we set to zero where it's outside (substracting 1) and set to
    # 1 when it's inside (dividing by 2). The rest is for practical 
    # manipulation of the tensor for later in the algorithm. 
    train_classes = train_input.sub(0.5).pow(2).sum(1).sub(R**2).sign().sub(1).div(-2).long().resize_((nb_samples,1))
    test_classes = test_input.sub(0.5).pow(2).sum(1).sub(R**2).sign().sub(1).div(-2).long().resize_((nb_samples,1))
    
    return train_input, train_classes, test_input, test_classes

# -------------------------------------------------------------------------- #

# This function just gives a set of n test_input and test_classes, this will
# allows us to do an average test error on the trained model, to be sure that 
# the error is significant.
def get_tests(n):
    M = []
    for k in range (n):
        L = []
        _, _, test_input, test_classes =  create_problem(1000)
        L.append(test_input)
        L.append(test_classes)
        M.append(L)
    return M