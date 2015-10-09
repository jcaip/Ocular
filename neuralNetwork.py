import scipy.io as spi
import scipy.optimize as spo
import numpy as np

class NeuralNetwork:
    X =0
    y=0
    tlarge=0
    input_layer_size=0
    hidden_layer_size=0
    output_layer_size=0
    epsilon = 10
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def initializeNeuralNetwork(self):
        initialTheta1 = np.random.rand(self.input_layer_size+1,self.hidden_layer_size)
        initialTheta2 = np.random.rand(self.hidden_layer_size+1,self.output_layer_size)

        initialNNParams = self.rollParams(initialTheta1,initialTheta2) 
        return initialNNParams

    def rollParams(self,t1,t2):
        return np.concatenate([t1.reshape(t1.size,1),t2.reshape(t2.size,1)])
        
    def unrollParams(self,tlarge):
        t1index = (self.input_layer_size+1)*self.hidden_layer_size
        t1 = tlarge[:t1index].reshape(self.input_layer_size+1,self.hidden_layer_size)    
        t2 = tlarge[t1index:].reshape(self.hidden_layer_size+1,self.output_layer_size)    
        return(t1,t2)

    def costFunction(self, *args):
        fthis, X, y, tlarge, epsilon  = args
        Theta1 = self.unrollParams(fthis)[0]
        Theta2 = self.unrollParams(fthis)[1]

        m = X.shape[0] 
        J = 0
        
        Theta1_grad = np.zeros((Theta1.shape))
        Theta2_grad = np.zeros((Theta2.shape))
        
        sigmoidVectorized = np.vectorize(self.sigmoid)
        biasTerm = np.ones((m,1))

        A1 = np.concatenate([biasTerm,self.X],1)
        A2 = np.concatenate([biasTerm,sigmoidVectorized(np.dot(A1, Theta1))],1)
        A3 = sigmoidVectorized(np.dot(A2, Theta2))

        largeY = np.zeros((m,self.output_layer_size)) 
        for i in range(0,self.output_layer_size):
            y_value = (y==i).astype(int)
            largeY[:,[i]] = y_value
        J = np.sum(np.sum(-largeY * np.log(A3) - (1-largeY) * np.log(1-A3)))/m
        #TODO refactor into two functions - one that returns cost and one that returns the gradients
        Theta1_reg = (epsilon/(2*m))*np.sum(np.sum(np.power(Theta1[:,1:],2)))
        Theta2_reg = (epsilon/(2*m))*np.sum(np.sum(np.power(Theta2[:,1:],2)))
        
        J = J+Theta1_reg+Theta2_reg

        print(J)
        return J

    def costFunctionGradient(self, *args):
        print((args[0]))
        fthis, X, y, tlarge, epsilon  = args
        
        Theta1 = self.unrollParams(fthis)[0]
        Theta2 = self.unrollParams(fthis)[1]

        m = X.shape[0] 
        
        Theta1_grad = np.zeros((Theta1.shape))
        Theta2_grad = np.zeros((Theta2.shape))
        
        sigmoidVectorized = np.vectorize(self.sigmoid)
        biasTerm = np.ones((m,1))

        A1 = np.concatenate([biasTerm,X],1)
        A2 = np.concatenate([biasTerm,sigmoidVectorized(np.dot(A1, Theta1))],1)
        A3 = sigmoidVectorized(np.dot(A2, Theta2))

        largeY = np.zeros((m,self.output_layer_size)) 
        for i in range(0,self.output_layer_size):
            y_value = (y==i).astype(int)
            largeY[:,[i]] = y_value
        delta_3 = A3-largeY
        #print(Theta2.shape)
        #print(Theta1.shape)
        delta_2 = np.dot(delta_3,Theta2[1:,:].transpose()) * sigmoidVectorized(np.dot(A1,Theta1))
        #print(delta_3.shape)
        #print(delta_2.shape)
        #print(A2.shape) 
        #print(A1.shape)
        Theta2_grad = (np.dot(A2.transpose(),delta_3))/m
        Theta1_grad = (np.dot(A1.transpose(),delta_2))/m
        
        #print(Theta2_grad.shape)
        #print(Theta1_grad.shape)
        Theta1_grad[1:,:] += epsilon/m*Theta1[1:,:]
        Theta2_grad[1:,:] += epsilon/m*Theta2[1:,:]
        return np.concatenate([np.ndarray.flatten(Theta1_grad),np.ndarray.flatten(Theta2_grad)])

    def predict(self,Theta1, Theta2, X): 
        num_labels = self.output_layer_size 
        m = self.X.shape[0]

        sigmoidVectorized = np.vectorize(sigmoid)
        biasTerm = np.ones((m,1))
        
        h1 = sigmoidVectorized(np.dot(np.concatenate([biasTerm,X],1), Theta1))
        h2 = sigmoidVectorized(np.dot(np.concatenate([biasTerm,h1],1), Theta2))
        return h2

    def train(self):
        #print(self.costFunctionGradient().shape)
        #print(self.tlarge.shape)
        args = (self.X, self.y, self.tlarge, self.epsilon)
        result = spo.fmin_cg(self.costFunction,np.ndarray.flatten(self.tlarge), fprime=self.costFunctionGradient, args=args)
        return result

    def __init__(self,b,c,trainingset,labels):
        self.input_layer_size = trainingset.shape[1]
        self.hidden_layer_size = b
        self.output_layer_size = c
        self.X =trainingset
        self.y = labels
        self.tlarge = self.initializeNeuralNetwork()

testSet = spi.loadmat('testweights.m')
tsd = testSet['X']
tsl = testSet['y']
nnTest = NeuralNetwork(50,10,tsd,tsl);
print(nnTest.train())

