from helpers import dlc_practical_prologue as prologue
import matplotlib.pyplot as plt
import matplotlib
from torch import optim
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable
import torch

##################################### Simple Conv Net
# ********************************* Function to compute errors for Simple Conv Net
def compute_nb_errors_simConvNet(model, data_input, data_target, mini_batch_size):
    nb_data_errors = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if data_target.data[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors

# *********************************  Function to train the model for Simple Conv Net
def train_model_simConvnet(model, train_input, train_target, validation_input, validation_target, device, nb_epochs, mini_batch_size, print_step):
    print("*************************Starting to train Archi1_SimpleConvNet")
    # Criterion to use on the training
    criterion = nn.CrossEntropyLoss()
    # Put criterion on GPU/CPU
    criterion.to(device)
    
    # Optimizer to use on the model 
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    # Vectors to accumulate the losses and accuracies
    training_loss = []
    training_accuracy = []
    validation_accuracy = []

    for e in range(nb_epochs+1):
        
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if e % print_step ==0 :
            training_loss.append(loss.item())
            print(f'\nEpoc : {e}, Loss: {loss.item()}')
            
            model.eval()

            training_accuracy.append(compute_nb_errors_simConvNet(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100)
            print(f'training_accuracy : {100-training_accuracy[-1]}')
                  
            validation_accuracy.append(\
                compute_nb_errors_simConvNet(model, validation_input, validation_target, mini_batch_size) / (validation_input.size(0)) * 100)
            print(f'Validation_accuracy : {100-validation_accuracy[-1]}')
            
            model.train()
            
    return model, training_loss, training_accuracy, validation_accuracy

# ********************************* Evaluate the Model Simple Conv Net
def eval_Model_simConvnet(mini_batch_size, nb_epochs, print_step, hidden_layers, acumulated_loss, model_arch):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("using cuda!")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    model = model_arch()
    print(model)
    model.to(device)
    # ********************************* Generate Data and Standarize
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)
    mean, std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    #Divide in validation and train input
    validation_input = train_input[900:1000]
    train_input = train_input[0:900]
    validation_target = train_target[900:1000]
    train_target = train_target[0:900]
    validation_clases = train_classes[900:1000]
    train_classes = train_classes[0:900]

    # Sent to variable so that can be run with autograd
    train_input, train_target, train_classes = Variable(train_input), Variable(train_target), Variable(train_classes)
    validation_input, validation_target, validation_clases = Variable(validation_input), Variable(validation_target), \
                                                            Variable(validation_clases)
    test_input, test_target, test_classes = Variable(test_input), Variable(test_target), Variable(test_classes)

    # Sent to device (Cuda if available)
    train_input, train_target, train_classes = train_input.to(device), train_target.to(device), train_classes.to(device)
    validation_input, validation_target, validation_clases = validation_input.to(device), validation_target.to(device), validation_clases.to(device)
    test_input, test_target, test_classes = test_input.to(device), test_target.to(device), test_classes.to(device)

    # Train model
    model, training_loss, training_accuracy, validation_accuracy  = train_model_simConvnet(model, train_input, \
                                                                train_target, validation_input, validation_target, device, nb_epochs, mini_batch_size, print_step)
    # *********************************  Show plots
    print("*************************Starting to generate plots for Archi1_SimpleConvNet")
    nb_epocs_array = []
    for epoc in range(0, nb_epochs+1, print_step):
        nb_epocs_array.append(epoc)
    validation_accuracy_plot =  [100 - int(x) for x in validation_accuracy] 
    fig, ax1 = plt.subplots()
    ax1.plot(nb_epocs_array, training_loss, 'g-')
    ax1.set_xlabel('Number of Epocs')
    ax1.set_ylabel('Loss', color='g')
    ax2 = ax1.twinx()
    ax2.plot(nb_epocs_array, validation_accuracy_plot, 'b-')
    ax2.set_ylabel('Accuracy Validation', color='b')
    plt.grid()
    plt.title("Archi 1 SimpleConvNet")
    plt.savefig('figures/loss-accuracyvsnumberepocs_arch1.png')
    plt.savefig('figures/loss-accuracyvsnumberepocs_arch1.pdf')
    # plt.show()

    # ********************************* Evaluate the model
    model.eval()
    print("*************************Evaluating the Archi1_SimpleConvNet")
    print(f"Error in training set: {compute_nb_errors_simConvNet(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100} %")
    print(f"Error in test set: {compute_nb_errors_simConvNet(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100} %")

    return compute_nb_errors_simConvNet(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100




##################################### Siammesse Network
# ********************************* Function to compute errors for Siammesse Network
def compute_nb_errors_comparator_Siammesse(model, data_input, data_target, mini_batch_size):
    """This function is to compute the error on predicting the clases 
    Input: Model
           Data_input = tensor of n*14*14
           Data_target = tesnor n*1 (with values either 0 or 1 depending on comparison of clases)

    Output:
           nb_data_errors: output the amount of times we missclassified a value
           after the result of comparison (0,1 depending on which value is greater)
    """
    nb_data_errors = 0
    test_a = []
    test_b = []
    result = []
    for b in range(0, data_input.size(0), mini_batch_size):
        out_x1, out_x2 = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes_x1 = torch.max(out_x1.data, 1)
        _, predicted_classes_x2 = torch.max(out_x2.data, 1)
        for k in range(mini_batch_size):
            test_a.append(predicted_classes_x1[k])
            test_b.append(predicted_classes_x2[k])
    for x in range(len(data_target)):
        if(test_a[x] > test_b[x]):
            result.append(0)
        else:
            result.append(1)

        if data_target.data[x] != result[x]:
            nb_data_errors = nb_data_errors + 1

    return nb_data_errors

# *********************************  Function to train the model for Siammesse Network
def train_model_Siammesse(model, train_input, train_target, train_classes, validation_input,
                validation_target, acumulated_loss, device, nb_epochs, mini_batch_size, print_step):
    # Criterion to use on the training
    criterion = nn.CrossEntropyLoss()

    # Put criterion on GPU/CPU
    criterion.to(device)

    # Optimizer to use on the model
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Vectors to accumulate the losses and accuracies
    training_loss = []
    training_accuracy_comparator = []
    validation_accuracy_comparator = []
    beta = 0

    for e in range(nb_epochs+1):
        if acumulated_loss:
            beta = (nb_epochs - e)/nb_epochs
        for b in range(0, train_input.size(0), mini_batch_size):
            x1, x2 = model(train_input.narrow(0, b, mini_batch_size))
            x1_loss = criterion(
                x1, train_classes[:, 0].narrow(0, b, mini_batch_size))
            x2_loss = criterion(
                x2, train_classes[:, 1].narrow(0, b, mini_batch_size))
            loss = (1 - beta)*x1_loss + beta*x2_loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if e % print_step == 0:
            training_loss.append(loss.item())
            print(f'\nEpoc : {e}, Loss: {loss.item()}')

            model.eval()

            training_accuracy_comparator.append(compute_nb_errors_comparator_Siammesse(
                model, train_input, train_target,mini_batch_size) / (train_input.size(0)/2) * 100)
            print(f'training_accuracy_comparator : {100-training_accuracy_comparator[-1]}')

            validation_accuracy_comparator.append(compute_nb_errors_comparator_Siammesse(
                model, validation_input, validation_target,mini_batch_size) / (validation_input.size(0)/2) * 100)
            print(f'validation_accuracy_comparator : {100-validation_accuracy_comparator[-1]}')

            model.train()

    return model, training_loss, training_accuracy_comparator, validation_accuracy_comparator

# ********************************* Evaluate the Model for Siammesse Network
def eval_Model_Siammesse(mini_batch_size, nb_epochs, print_step, hidden_layers, acumulated_loss, model_arch, num_arch ):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("using cuda!")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    model = model_arch(hidden_layers)
    print(model)
    model.to(device)

    # ********************************* Generate Data and Standarize
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
        1000)
    mean, std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    # Divide in validation and train input
    validation_input = train_input[900:1000]
    train_input = train_input[0:900]
    validation_target = train_target[900:1000]
    train_target = train_target[0:900]
    validation_clases = train_classes[900:1000]
    train_classes = train_classes[0:900]

    # Sent to variable so that can be run with autograd
    train_input, train_target, train_classes = Variable(
        train_input), Variable(train_target), Variable(train_classes)
    validation_input, validation_target, validation_clases = Variable(validation_input), Variable(validation_target), \
        Variable(validation_clases)
    test_input, test_target, test_classes = Variable(
        test_input), Variable(test_target), Variable(test_classes)

    # Sent to device (Cuda if available)
    train_input, train_target, train_classes = train_input.to(
        device), train_target.to(device), train_classes.to(device)
    validation_input, validation_target, validation_clases = validation_input.to(
        device), validation_target.to(device), validation_clases.to(device)
    test_input, test_target, test_classes = test_input.to(
        device), test_target.to(device), test_classes.to(device)

    # Train model
    model, training_loss, training_accuracy_comparator, validation_accuracy_comparator = train_model_Siammesse(
        model, train_input, train_target, train_classes, validation_input, validation_target,  acumulated_loss,  device, nb_epochs, mini_batch_size, print_step)

    # *********************************  Show plots
    if num_arch == 2:
        print("*************************Starting to generate plots for Arch2 ConvNetSiam_noWS_noDr_noBN_2")
    elif num_arch == 3:
        print("*************************Starting to generate plots for Arch3 ConvNetSiam_WS_noDr_noBN_3")
    else:
        print("*************************Starting to generate plots for Arch4 ConvNetSiam_WS_Dr_BN")

    nb_epocs_array = []
    for epoc in range(0, nb_epochs+1, print_step):
        nb_epocs_array.append(epoc)
    validation_accuracy_comparator_plot =  [100 - int(x) for x in validation_accuracy_comparator] 
    fig, ax1 = plt.subplots()
    ax1.plot(nb_epocs_array, training_loss, 'g-')
    ax1.set_xlabel('Number of Epocs')
    ax1.set_ylabel('Loss', color='g')
    ax2 = ax1.twinx()
    ax2.plot(nb_epocs_array, validation_accuracy_comparator_plot, 'b-')
    ax2.set_ylabel('Accuracy Validation', color='b')
    plt.grid()
    if num_arch == 2:
        plt.title("Arch 2 - ConvNetSiam_noWS_noDr_noBN_2")
        plt.savefig('figures/loss-accuracyvsnumberepocs_arch2.png')
        plt.savefig('figures/loss-accuracyvsnumberepocs_arch2.pdf')
    elif num_arch == 3:
        plt.title("Arch 3 - ConvNetSiam_WS_noDr_noBN_3")
        plt.savefig('figures/loss-accuracyvsnumberepocs_arch3.png')
        plt.savefig('figures/loss-accuracyvsnumberepocs_arch3.pdf')
    else: 
        plt.title("Arch 4 - ConvNetSiam_WS_Dr_BN")
        plt.savefig('figures/loss-accuracyvsnumberepocs_arch4.png')
        plt.savefig('figures/loss-accuracyvsnumberepocs_arch4.pdf')
    # plt.show()

    # ********************************* Evaluate the model
    model.eval()
    if num_arch == 2:
        print("*************************Evaluating the Arch 2 ConvNetSiam_noWS_noDr_noBN_2")
    elif num_arch == 3:
        print("*************************Evaluating the Arch 3 ConvNetSiam_WS_noDr_noBN_3")
    else:
        print("*************************Evaluating the Arch 4 ConvNetSiam_WS_Dr_BN")
    
    print(f"Error in training predicting result of comparator is: {compute_nb_errors_comparator_Siammesse(model, train_input, train_target, mini_batch_size) / (train_input.size(0)/2) * 100} %")
    print(f"Error in test set predicting  result of comparator is: {compute_nb_errors_comparator_Siammesse(model, test_input, test_target, mini_batch_size) / (test_input.size(0)/2) * 100} %")
    return compute_nb_errors_comparator_Siammesse(model, test_input, test_target, mini_batch_size) / (test_input.size(0)/2) * 100



##################################### Adv Conv Net
# ********************************* Function to compute errors for Adv Conv Net
def compute_nb_errors_comparator_AdvConvNet(model, data_input, data_target, mini_batch_size):
    """This function is to compute the error on predicting the clases 
    Input: Model
           Data_input = tensor of n*14*14
           Data_target = tesnor n*1 (with values either 0 or 1 depending on comparison of clases)
    
    Output:
           nb_data_errors: output the amount of times we missclassified a value
           after the result of comparison (0,1 depending on which value is greater)
    """
    nb_data_errors = 0
    test_a = []
    test_b = []
    result = []
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output.data, 1)
        for k in range(mini_batch_size):
            if (k%2 ==0):
                test_a.append(predicted_classes[k])
            else:
                test_b.append(predicted_classes[k])
            
    for x in range(len(data_target)):
        if(test_a[x] > test_b[x]):
            result.append(0)
        else:
            result.append(1)

        if data_target.data[x] != result[x]:
            nb_data_errors = nb_data_errors + 1

    return nb_data_errors


# *********************************  Function to train the model for Adv Conv Net
def train_model_AdvConvNet(model, train_input, train_target, train_classes, validation_input,\
                validation_target, validation_classes, device, nb_epochs, mini_batch_size, print_step):
    # Criterion to use on the training
    criterion = nn.CrossEntropyLoss()
    
    # Put criterion on GPU/CPU
    criterion.to(device)
    
    # Optimizer to use on the model 
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    # Vectors to accumulate the losses and accuracies
    training_loss = []
    training_accuracy_comparator = []
    validation_accuracy_comparator = []
        
    for e in range(nb_epochs+1):
        
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_classes.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()
        if e % print_step ==0 :            
            training_loss.append(loss.item())
            print(f'\nEpoc : {e}, Loss: {loss.item()}')
            
            model.eval()

           
            training_accuracy_comparator.append(\
                compute_nb_errors_comparator_AdvConvNet(model, train_input, train_target, mini_batch_size) / (train_input.size(0)/2) * 100)
            print(f'training_accuracy_comparator : {100-training_accuracy_comparator[-1]}')
                  
                              
            validation_accuracy_comparator.append(\
                compute_nb_errors_comparator_AdvConvNet(model, validation_input, validation_target, mini_batch_size) / (validation_input.size(0)/2) * 100)
            print(f'validation_accuracy_comparator : {100-validation_accuracy_comparator[-1]}')
            
            model.train()
            
    return model, training_loss, training_accuracy_comparator, validation_accuracy_comparator

# ********************************* Evaluate the Model
def eval_Model_AdvConvNet(mini_batch_size, nb_epochs, print_step, hidden_layers, acumulated_loss, model_arch ):
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("using cuda!")
    else:
        device = torch.device('cpu')
        print("Using CPU")
   
    model = model_arch(hidden_layers)
    print(model)
    model.to(device)

    # ********************************* Generate Data and Standarize
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
        1000)
    mean, std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    # Divide in validation and train input
    train_input = train_input.view(-1, 1, 14, 14)
    validation_input = train_input[1800:2000]
    train_input = train_input[:1800]
    train_classes = train_classes.view(-1, 1)
    validation_classes = train_classes[1800:2000]
    train_classes = train_classes[:1800]
    validation_target = train_target[900:1000]
    train_target = train_target[:900]
    test_input = test_input.view(-1, 1, 14, 14)
    test_classes = test_classes.view(-1, 1)
    test_classes = test_classes.reshape((-1,))
    train_classes = train_classes.reshape((-1,))
    validation_classes = validation_classes.reshape((-1,))

    # Sent to variable so that can be run with autograd
    train_input, train_target, train_classes = \
        Variable(train_input), Variable(train_target), Variable(train_classes)
    test_input, test_target, test_classes = \
        Variable(test_input), Variable(test_target), Variable(test_classes)
    validation_input, validation_target, validation_classes = \
        Variable(validation_input), Variable(
            validation_target), Variable(validation_classes)

    # Sent to device (Cuda if available)
    train_input, train_target, train_classes = \
        train_input.to(device), train_target.to(
            device), train_classes.to(device)
    test_input, test_target, test_classes = \
        test_input.to(device), test_target.to(device), test_classes.to(device)
    validation_input, validation_target, validation_classes = \
        validation_input.to(device), validation_target.to(
            device), validation_classes.to(device)
    # Train model
    model, training_loss, training_accuracy_comparator, validation_accuracy_comparator = train_model_AdvConvNet(model, train_input, train_target, train_classes, validation_input,\
                validation_target, validation_classes,  device, nb_epochs, mini_batch_size, print_step)
    
    # *********************************  Show plots
    print("*************************Starting to generate plots for Archi5_advancedConvNet")
    nb_epocs_array = []
    for epoc in range(0, nb_epochs+1, print_step):
        nb_epocs_array.append(epoc)
    validation_accuracy_comparator_plot = [100 - int(x) for x in validation_accuracy_comparator]
    fig, ax1 = plt.subplots()
    ax1.plot(nb_epocs_array, training_loss, 'g-')
    ax1.set_xlabel('Number of Epocs')
    ax1.set_ylabel('Loss', color='g')
    ax2 = ax1.twinx()
    ax2.plot(nb_epocs_array, validation_accuracy_comparator_plot, 'b-')
    ax2.set_ylabel('Accuracy Validation', color='b')
    plt.grid()
    plt.title("Archi 5 advancedConvNet")
    plt.savefig('figures/loss-accuracyvsnumberepocs_arch5.png')
    plt.savefig('figures/loss-accuracyvsnumberepocs_arch5.pdf')
    # plt.show()

    # ********************************* Evaluate the model
    model.eval()
    print("*************************Evaluating the Archi5_advancedConvNet")
    print(f"Error in training set: {compute_nb_errors_comparator_AdvConvNet(model, train_input, train_target, mini_batch_size) / (train_input.size(0)/2) * 100} %")
    print(f"Error in test set: {compute_nb_errors_comparator_AdvConvNet(model, test_input, test_target, mini_batch_size) / (test_input.size(0)/2) * 100} %")
    return compute_nb_errors_comparator_AdvConvNet(model, test_input, test_target, mini_batch_size) / (test_input.size(0)/2) * 100
