import Function 
from Function import *
import matplotlib.pyplot as plt

train_input, train_target = generate_disc_set(1000);
test_input, test_target = generate_disc_set(1000);

mean, std = train_input.mean(), train_input.std();

train_input.sub_(mean).div_(std);
test_input.sub_(mean).div_(std);

# train_input, train_target
nb_epochs=100
mini_batch_size=1
Model = Sequential([Linear(2,128),ReLu(),Linear(128,2),ReLu()], LossMSE()) #Building the model and linking the loss
Model.set_Lr(0.007) #Setting the learning rate
# Model.print() #Print Parameters of the model
print("Before training, Train Error: {:.4f}%, Test Error: {:.4f}%".format(compute_nb_errors(Model,train_input,train_target,mini_batch_size)/train_input.size(0)*100,compute_nb_errors(Model,test_input, test_target,mini_batch_size)/test_input.size(0)*100))
train_target_one_hot= convert_to_one_hot_labels(train_input,train_target)
History_Loss=[]
for epochs in range(0,nb_epochs):
    Sum_Loss=0
    for b in range(0, train_input.size(0), 1):
        output = Model.forward(train_input.narrow(0, b, 1))
        Loss= Model.backward(train_target_one_hot.narrow(0, b, 1),output)
        Sum_Loss=Sum_Loss+Loss.item()
    History_Loss.append(Sum_Loss)  
    #print("Epoch: {}, Train Error: {:.4f}%, Test Error: {:.4f}%, Loss  {:.4f}".format(epochs+1,compute_nb_errors(Model,train_input, train_target,1)/train_input.size(0)*100,compute_nb_errors(Model,test_input, test_target,1)/test_input.size(0)*100,Sum_Loss))

        
print("After training, Train Error: {:.4f}%, Test Error: {:.4f}%, Loss  {:.4f}".format(compute_nb_errors(Model,train_input,train_target,mini_batch_size)/train_input.size(0)*100,compute_nb_errors(Model,test_input, test_target,mini_batch_size)/test_input.size(0)*100,Sum_Loss))

    