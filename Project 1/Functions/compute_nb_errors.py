# Import of troch library
import torch

# Function for computing the number of error with the target
def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        # Evaluation of the trained model
        model.eval()
        _, _, result = model(data_input.narrow(0, b, mini_batch_size))
        model.train()
        
        _, predicted_classes = torch.max(result, 1)
        
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors