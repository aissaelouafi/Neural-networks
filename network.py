import random
import numpy as np
class Network(object):

    """
    Intialize the nural network object by choosing randomly
    biases and weights parameters
    """
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y,1) for y in sizes[1:]] """ Initialize biases randomly using a gaussian distributions with mean 0 and standard deviation 1 """
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]


    """
    Define the sigmoid function, it can be applied to an np.array object
    """
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    """
    The sigmoid derivation of the sigmoid function
    """
    def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z))

   """
   Feed forward function take an input a and return the otput based on sigmoid function
   """
    def feedforward(self,x):
        for b,w in zip(self.biases,self.weights):
            x = sigmoid(np.dot(w,x)+b)
        return x

    """
    Train the NN using the mini-batch stochastic gradient descent algorithm
    training data is a list of tuples (x,y) representing the trainings input
    with desired output. test_data is optional argument if it provided the SGD
    will evaluate the network after each epoch and print the progress.
    """
    def SGD(self,training_data,epochs,mini_batch_size,lr,test_data=None):
        if test_data : n_test = len(test_data)
        n = len(training_data)
        for j in epochs:
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,lr)
            if test_data:
                print "Epoch {0}: {1}/{2}".format(j,self.evaluate(test_data),n_test)
            else:
                print "Epoch {0} complete".format(j)

    def backpropagation(self,x,y):
        nabla_b = [np.zeros(b.shape) for b in self.biases] #get the nabla_b layer by layer
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        #feedforward
        activation = x
        activations = [x] #store activations layer by layer
        for b,w
