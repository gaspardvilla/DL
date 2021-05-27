# Import of troch library
import torch
from torch import optim
from torch import nn

# Import other functions
from Functions.compute_nb_errors import compute_nb_errors

# Function that train and test the model
def train_and_test_model(Models, train_input, train_target, train_classes, Tests, nb_epochs, mini_batch_size):
    
    # Initialization for the optimization
    criterion = nn.CrossEntropyLoss()
    eta = 1e-3
    loss_coeff = 10
    
    # Initialization of the train and test error list
    Train_error = []
    Test_error = []
    std_deviation = 0.0 # We only need the standard deviation for the last step so no need of list
    
    for e in range(nb_epochs):
        
        train_error = 0.0
        avg_nb_test_error = torch.tensor(())
        
        # Print of the actual epoch
        print('epoch =', e)
        
        for k in range(len(Models)):
            
            model = Models[k]
            optimizer = optim.Adam(model.parameters(), lr = eta)
            
            # Batch
            for b in range(0, train_input.size(0), mini_batch_size):
                digit1, digit2, result = model(train_input.narrow(0, b, mini_batch_size))
                    
                loss_result = criterion(result, train_target.narrow(0, b, mini_batch_size))
                loss_digit1 = criterion(digit1, train_classes[:,0].narrow(0, b, mini_batch_size))
                loss_digit2 = criterion(digit2, train_classes[:,1].narrow(0, b, mini_batch_size))
                
                if model.AL:
                    loss = loss_result + loss_coeff*loss_digit1 + loss_coeff*loss_digit2
                else:
                    loss = loss_result
                
                model.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Compute train and test error for e epochs
            nb_train_errors = compute_nb_errors(model, train_input, train_target, mini_batch_size)
            train_error += nb_train_errors / 10
            
            nb_test_errors = compute_nb_errors(model, Tests[k][0], Tests[k][1], mini_batch_size)
            nb_test_errors = torch.tensor([nb_test_errors/10]).float()
            avg_nb_test_error = torch.cat((avg_nb_test_error, nb_test_errors), 0)
        
        Train_error.append(train_error / len(Models))
        Test_error.append(avg_nb_test_error.mean().tolist())
        std_deviation = avg_nb_test_error.std().tolist()

    return Train_error, Test_error, std_deviation


