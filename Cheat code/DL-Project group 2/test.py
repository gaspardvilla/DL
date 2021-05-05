from architectures import simpleConvNet_1 as simpleConvNet
from architectures import ConvNetSiam_noWS_noDr_noBN_2 as ConvNetSiam_noWS_noDr_noBN
from architectures import ConvNetSiam_WS_noDr_noBN_3 as ConvNetSiam_WS_noDr_noBN
from architectures import ConvNetSiam_WS_Dr_BN_4 as ConvNetSiam_WS_Dr_BN
from architectures import advancedConvNet_5 as advancedConvNet
from helpers import helpers as helpers
# ********************************* Define Parameters
mini_batch_size = 100
nb_epochs = 250
print_step = 50
hidden_layers = 256
acumulated_loss = True



if __name__ == "__main__":
    print(' ################ Running all the networks, please wait until is finished')
    # Model 1: simpleConvNet
    print(' \n\n\n\n####### Running Arch1 Simple Conv Net')
    error_arch1 = helpers.eval_Model_simConvnet(mini_batch_size, nb_epochs, print_step, hidden_layers, acumulated_loss, simpleConvNet.SimpleConvNet)
    # Model 2: Siammesse Network ConvNet No WS no Dropout no Batch Norm
    print(' \n\n\n\n####### Running Arch2 ConvNetSiam_noWS_noDr_noBN')
    error_arch2 = helpers.eval_Model_Siammesse(mini_batch_size, nb_epochs, print_step, hidden_layers, acumulated_loss, ConvNetSiam_noWS_noDr_noBN.ConvNetSiam_noWS_noDr_noBN,2)
    # Model 3: Siammesse Network ConvNet No WS no Dropout no Batch Norm
    print(' \n\n\n\n####### Running Arch3 ConvNetSiam_WS_noDr_noBN')
    error_arch3 = helpers.eval_Model_Siammesse(mini_batch_size, nb_epochs, print_step, hidden_layers, acumulated_loss, ConvNetSiam_WS_noDr_noBN.ConvNetSiam_WS_noDr_noBN,3)
    # Model 4: Siammesse Network Conv Net with Dropout with Batch Norm
    print(' \n\n\n\n####### Running Arch4 ConvNetSiam_WS_Dr_BN')
    error_arch4 = helpers.eval_Model_Siammesse(mini_batch_size, nb_epochs, print_step, hidden_layers, acumulated_loss, ConvNetSiam_WS_Dr_BN.ConvNetSiam_WS_Dr_BN,4)
    # Model 5: Advance ConvNet with Dropout and Batch Norm
    print(' \n\n\n\n####### Running Arch5 Advanced Conv Net')
    error_arch5 = helpers.eval_Model_AdvConvNet(mini_batch_size, nb_epochs, print_step, hidden_layers, acumulated_loss, advancedConvNet.advancedConvNet)

    print("\n\n\n\n####### The accuracy in the test sets are: \n")
    print("Arch 1 simpleConvNet = ", 100.0-error_arch1)
    print("Arch 2 ConvNetSiam_noWS_noDr_noBN = ", 100.0-error_arch2)
    print("Arch 3 ConvNetSiam_WS_noDr_noBN = ", 100.0-error_arch3)
    print("Arch 4 ConvNetSiam_WS_Dr_BN = ", 100.0-error_arch4)
    print("Arch 5 advancedConvNet = ", 100.0-error_arch5)
