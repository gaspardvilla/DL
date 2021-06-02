# This function do a simple normalization of the digits by dividing by the
# maximum value that a pixel can take. In our case, we already know that the
# maximum value is equal to 255. 
def digit_normalization(train_input):
    train_input = train_input / 255
    return train_input