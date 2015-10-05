import scipy as sp
import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))

input_layer_size=0
hidden_layer_size=0
output_layer_size=0

def initializeNeuralNetwork(a,b,c):
    input_layer_size = a
    hidden_layer_size = b
    output_layer_size = c

    initialTheta1 = np.random.rand(input_layer_size,hidden_layer_size)
    initialTheta2 = np.random.rand(hidden_layer_size,output_layer_size)

    initialNNParams = rollParams(initialTheta1,initialTheta2) #TODO figure out function unrolling/rolling
    return initialNNParams

def rollParams(t1,t2):
    np.concatenate([t1.reshape(t1.size,1),t2.reshape(t2.size,1)])
    
def unrollParams(tlarge):
    t1index = input_layer_size*hidden_layer_size
    t1 = tlarge[:t1index].reshape(input_layer_size,hidden_layer_size)    
    t2 = tlarge[t1index:].reshape(hidden_layer_size,output_layer_size)    
    return(t1,t2)

def feedForward(Theta1, Theta2, X): #where Theta1 and Theta2 are hypothesis arrays and X is the training set
    m = input_layer_size
    num_labels = output_layer_size 

    p = np.zeros(m,1)

    sigmoidVectorized = np.vectorize(sigmoid)
    
    biasTerm = np.ones((m,1))

    h1 = sigmoidVectorized(np.dot(np.concatenate([biasTerm,X],1), Theta1))
    h2 = sigmoidVectorized(np.dot(np.concatenate([biasTerm,h1],1), Theta2))
    p = 

initializeNeuralNetwork(400,200,86)

