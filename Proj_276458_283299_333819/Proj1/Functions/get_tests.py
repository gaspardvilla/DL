# Give for the project
import dlc_practical_prologue as prologue
from Functions.digit_normalization import digit_normalization


# Function that return a list of n test sets of 1000 pair sets from MNIST
def get_tests(n):
    
    # Initialization of the list of test sets
    M = []
    
    for k in range (0, n):
        L = []
        _, _, _, test_input, test_target, test_classes \
                                        =  prologue.generate_pair_sets(1000)
        
        # Normalization of the test sets
        test_input = digit_normalization(test_input)
        L.append(test_input)
        L.append(test_target)
        L.append(test_classes)
        M.append(L)
        
    return M