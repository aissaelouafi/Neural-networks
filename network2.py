from random import seed
from random import random
from math import exp

#Initialize the neural network with the number of inputs, the number of neurons in the hidden layer and n_outputs in the output layer
def initialize_netwrok(n_inputs,n_hidden,n_outputs):
    network = list() #W'll implement network as a list of layer and layer as a list of dictionnary (weights and bias)
    hidden_layers = [{'weights ': [random() for i in range(n_inputs+1)]} for i in range(n_hidden)] #randomly Initialize the weights of hidden layer, in this case the bias is a special weight
    network.append(hidden_layers)
    output_layer = [{'weights ':[random() for i in range(n_outputs+1)]} for i in range(n_outputs)] # randomy initialize the weights of output layyer of the neural networkds
    network.append(output_layer)
    return network

#We will now calculate the neuron activation of one neuron given an input that could be a row from the dataset
#Neuron activation is calculates as the weighted sum of weight : ctivation = sum(weight_i*input-i) + bias
#We can say that the bias is the last term of the weights list
def activate(weights,inputs):
    activation = weights[-1] #bias term
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
    return activation

#Once the neuron is activated, we need to transfert the activation to see what the neuron output actually is
#Different functions can be used (sigmoid function, tangh and rectifier transfert function), We'll use the sigmoid function
#It can take any input and produces a value between 0 and 1, we can easily calculate the derivation
# sigmoid function is output = 1/(1+exp(-activation))
def transfert(activation):
    return 1.0/1.0+exp(-activation)

#Forwad propagation : we wor through each layer of our network calculating the outputs for each neurons. Outputs from one layer becomes
#input to the neurons of the following layers
def forward_propagate(network,row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'],inputs)
            neuron['output'] = transfert(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def transfert_derivate(output):
    return output * (1 - output)
    
if __name__ == "__main__":
    seed(1)
    network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],[{'weights': [0.2550690257394217, 0.49543508709194095]}, {'weights': [0.4494910647887381, 0.651592972722763]}]]
    row = [1,0,None]
    output = forward_propagate(network,row)
    print(output)