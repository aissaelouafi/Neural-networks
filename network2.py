from random import seed
from random import random
from math import exp

#Initialize the neural network with the number of inputs, the number of neurons in the hidden layer and n_outputs in the output layer
def initialize_netwrok(n_inputs,n_hidden,n_outputs):
    network = list() #W'll implement network as a list of layer and layer as a list of dictionnary (weights and bias)
    hidden_layers = [{'weights': [random() for i in range(n_inputs+1)]} for i in range(n_hidden)] #randomly Initialize the weights of hidden layer, in this case the bias is a special weight (the last index of the weights array)
    network.append(hidden_layers)
    output_layer = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)] # randomly initialize the weights of output layyer of the neural networkds
    network.append(output_layer)
    return network

#We will now calculate the neuron activation of one neuron given an input that could be a row from the dataset
#Neuron activation is calculates as the weighted sum of weight : activation = sum(weight_i*input_i) + bias
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
    return 1.0 / (1.0+exp(-activation))

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

#The derivate function of the transfert function, we use only the sigmoid function
def transfert_derivate(output):
    return output * (1 - output)

#Error propagation in order to calculate the error for each output neuron that give an error signal signal (input) to propagate backwards
#through the netwrok, the error for a given output neuron can be calculated as follows : error = (excepted-output)*transfert_derivate(output)
#the output is the value predicted by the output neuron and excpeted is the desired output value

#The error signal of a hidden layer is more complicated, the error signal for a neuron in the hidden layer is calculated as the weighted error
#of each neuron in the output error = (weight_k * error_j) * transfer_derivative(output) error_j is the error calculated from the jth output neuron
#weight_k is the weight that connect the kth neuron

#The error signal in the hidden layer is accumulated from neurons in the ouput layer
def backward_propagate_error(network,expected):
    for i in reversed(range(len(network))): #loop on network reversed
        layer = network[i] #get current layer
        errors = list() #initialize errors
        if i != len(network) - 1: #calculcate the error of hidden layers
            for j in range(len(layer)): #loop on neurons in the hidden layers
                error = 0.0
                for neuron in network[i+1]: #get neurons in the hidden layer
                    error += (neuron['weights'][j]*neuron['delta'])
                errors.append(error)
        else : #output layer
            for j in range(len(layer)): #explore neuron in the output layer
                neuron = layer[j]
                errors.append(expected[j]-neuron['output'])
        for j in range(len(layer)): #explore neuron in layer
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfert_derivate(neuron['output'])


#The network is trained using a stochastic gradient descent, the involves multiple iterations on row dat => data forward propagating
#backpropagating the error and update the weights of our neural netwroks so we will update weights and train the neural networks
#Once error ar calculated for each neuron in the network via the back propagation method, they can used to update the netwroks weights
#using the following fomula : weight = weight + learning_rate * error *input (lr : parameter, error : error callculated by the BPE algorithm (in this case the delta value)
# and input is the input value caused the error), the same procedure can be applyed to bias (we can suppose that the weight of bias is 1)

#update weights of a given raw data input
#input of output layer is the output from the hidden layers
#TODO : Use stochasitic gradient descent (shuffle data and use only a part of data, it will be verry fast)
def update_weights(network,row,learning_rate):
    for i in range(len(network)): #loop on network layers
        inputs = row
        if i!= 0:
            inputs = [neuron['output'] for neuron in network[i-1]] #Get the neuron output from the neural networks
        for neuron in network[i]: #loop on neuron in the ith layer
            for j in range(len(inputs)-1): #loop on inout data
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['delta']


#Train network : looping for a fixed number of epochs and within each epoch updating the network for each row in the training dataset
#the execpted number of output values is used to transform class value in the training data set into a one hot encoder
#we can plot the evolution of error rate of each epoch iterations in order to see the evolution of networks based on epochs
#sum of squared error
def train_network(network,train,learning_rate,n_epochs,n_outputs):
    for epoch in range(n_epochs):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network,row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network,expected)
            update_weights(network,row,learning_rate)
        print('>> epoch = %d, learning_rate = %.3f, error = %.3f' % (epoch,learning_rate,sum_error))

#Define a function to predict the output of a trained neural networks
def predict(network,row):
    outputs = forward_propagate(network,row)
    return outputs.index(max(outputs))

if __name__ == "__main__":
    seed(1)
    # Backpropagation error :
    #network = [[{'output': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
    #		[{'output': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]},
    #        {'output': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
    #expected = [0, 1]
    #row = [1,0,None]
    #backward_propagate_error(network,expected)
    #for layer in network:
    #    print(layer)

    dataset = [[2.7810836,2.550537003,0],
	[1.465489372,2.362125076,0],
	[3.396561688,4.400293529,0],
	[1.38807019,1.850220317,0],
	[3.06407232,3.005305973,0],
	[7.627531214,2.759262235,1],
	[5.332441248,2.088626775,1],
	[6.922596716,1.77106367,1],
	[8.675418651,-0.242068655,1],
	[7.673756466,3.508563011,1]]


    #def initialize_netwrok(n_inputs,n_hidden,n_outputs):
    n_inputs = len(dataset[0])-1 #get the number of input (in this case we have only 2 outputs)
    n_outputs = len(set([row[-1] for row in dataset]))
    network = initialize_netwrok(n_inputs,2,n_outputs)
    #print(network)
    train_network(network,dataset,0.500,30,n_outputs)
    for row in dataset:
        prediction = predict(network,row)
        print('>Excepted = %d, Predicted = %d' % (row[-1],prediction))
    #Predict on a row data
    #test = [2.674382,2.45555423,0]
    #print(network)
