# Give for the project
import dlc_practical_prologue as prologue


# Function that return a list of n test sets of 1000 pair sets from MNIST
def get_tests(n):
    
    M = []
    
    for k in range (0, n):
        L = []
        _, _, _, test_input, test_target, test_classes \
                                        =  prologue.generate_pair_sets(1000)
        L.append(test_input)
        L.append(test_target)
        L.append(test_classes)
        M.append(L)
        
    return M