import scipt.optimize as spo
import numpy as np

class NeuralNetwork:
    X
    y
    tlarge
    input_layer_size
    hidden_layer_size
    output_layer_size 
    epsilon =1

    def __init__(a,b,c,trainingset,labels):
        global input_layer_size
        global hidden_layer_size 
        global output_layer_size
        global X
        global y
        global tlarge
        input_layer_size = a
        hidden_layer_size = b
        output_layer_size = c
        X =trainingset
        y = labels
        tlarge = initializeNeuralNetwork()

    def sigmoid(z):
        return 1/(1+np.exp(-z))

    def initializeNeuralNetwork():
        initialTheta1 = np.random.rand(input_layer_size+1,hidden_layer_size)
        initialTheta2 = np.random.rand(hidden_layer_size+1,output_layer_size)

        initialNNParams = rollParams(initialTheta1,initialTheta2) 
        return initialNNParams

    def rollParams(t1,t2):
        return np.concatenate([t1.reshape(t1.size,1),t2.reshape(t2.size,1)])
        
    def unrollParams(tlarge):
        t1index = (input_layer_size+1)*hidden_layer_size
        t1 = tlarge[:t1index].reshape(input_layer_size+1,hidden_layer_size)    
        t2 = tlarge[t1index:].reshape(hidden_layer_size+1,output_layer_size)    
        return(t1,t2)

    def costFunction():

        Theta1 = unrollParams(tlarge)[0]
        Theta2 = unrollParams(tlarge)[1]

        m = X.shape[0] 
        J = 0
        
        Theta1_grad = np.zeros((Theta1.shape))
        Theta2_grad = np.zeros((Theta2.shape))
        
        sigmoidVectorized = np.vectorize(sigmoid)
        biasTerm = np.ones((m,1))

        A1 = np.concatenate([biasTerm,X],1)
        A2 = np.concatenate([biasTerm,sigmoidVectorized(np.dot(A1, Theta1.transpose()))],1)
        A3 = sigmoidVectorized(np.dot(A2, Theta2.transpose()))

        largeY = np.zeros((m,output_layer_size)) 
        for i in range(0,output_layer_size):
            y_value = (y==i).astype(int)
            largeY[:,i] = y_value
        J = np.sum(np.sum(-largeY * np.log(A3) - (1-largeY) * np.log(1-A3)))/m
        #TODO refactor into two functions - one that returns cost and one that returns the gradients
        Theta1_reg = (epsilon/(2*m))*np.sum(np.sum(np.power(Theta1[:,1:],2)))
        Theta2_reg = (epsilon/(2*m))*np.sum(np.sum(np.power(Theta2[:,1:],2)))
        
        J = J+Theta1_reg+Theta2_reg


        return J

    def costFunctionGradient():
        Theta1 = unrollParams(tlarge)[0]
        Theta2 = unrollParams(tlarge)[1]

        m = X.shape[0] 
        
        Theta1_grad = np.zeros((Theta1.shape))
        Theta2_grad = np.zeros((Theta2.shape))
        
        sigmoidVectorized = np.vectorize(sigmoid)
        biasTerm = np.ones((m,1))

        A1 = np.concatenate([biasTerm,X],1)
        A2 = np.concatenate([biasTerm,sigmoidVectorized(np.dot(A1, Theta1.transpose()))],1)
        A3 = sigmoidVectorized(np.dot(A2, Theta2.transpose()))

        largeY = np.zeros((m,output_layer_size)) 
        for i in range(0,output_layer_size):
            y_value = (y==i).astype(int)
            largeY[:,i] = y_value
        delta_3 = A3-largeY
        delta_2 = np.dot(delta_3,Theta2[:,1:]) * sigmoidVectorized(np.dot(A1,Theta1.transpose()))

        Theta2_grad = (np.dot(delta_3.transpose(),A2)/m
        Theta1_grad = (np.dot(delta_2.transpose(),A1)/m

        Theta1_grad[:,1:] += epsilon/m*Theta1[1:1:]
        Theta2_grad[:,1:] += epsilon/m*Theta2[1:1:]
        return np.concatenate([Theta1_grad.reshape((Theta1_grad.size,1)),Theta2_grad.reshape((Theta2_grad.size,1))
    
    def predict(Theta1, Theta2, X): #where Theta1 and Theta2 are hypothesis arrays and X is the training set
        num_labels = output_layer_size 
        m = X.shape[0]

        sigmoidVectorized = np.vectorize(sigmoid)
        biasTerm = np.ones((m,1))
        
        h1 = sigmoidVectorized(np.dot(np.concatenate([biasTerm,X],1), Theta1))
        h2 = sigmoidVectorized(np.dot(np.concatenate([biasTerm,h1],1), Theta2))
        return h2

    def train():
        spo.fmin_cg(costFunction(), tlarge, )
