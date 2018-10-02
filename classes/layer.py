import numpy as np

class Layer():

    def __init__(self, weights, bias, activation, activationDerivative):

        self.weights = weights

        self.bias = bias

        # activation function G( Z[l] )
        self.activation = activation
        
        # derivative of the activation function G'( Z[l] , A[l] )
        self.activationDerivative = activationDerivative

    def setPreviousLayer(self, layer):
        self.previousLayer = layer

    def setNextLayer(self, layer):
        self.nextLayer = layer
    
    def getACacheValue(self):
        return self.ACacheValue
    
    def getWeightsGradient(self):
        return self.dW
    
    def getBiasGradient(self):
        return self.db

    def forwardPropTrain(self):
        #performs G( W*A[i-1] + b ), and caches values for backpropagation
        previousAValue = self.previousLayer.forwardPropTrain()
        self.ACacheValue = self.activation( np.dot( self.weights , previousAValue ) + self.bias )
        self.activationDerivativeCacheValue = self.activationDerivative(previousAValue, self.ACacheValue)
        return self.ACacheValue #may be optimized, on how data is accessed
    
    def forwardPropPredict(self):
        return self.activation( np.dot( self.weights, self.previousLayer.forwardPropPredict() ) + self.bias )


    def backwardProp(self,m):
        # performs backpropagation

        dZ = np.multiply( self.nextLayer.backwardProp(m) , self.activationDerivativeCacheValue )

        self.dW = (1/m)*np.dot( dZ , np.transpose( self.previousLayer.getACacheValue() ) )

        self.db = (1/m)*np.sum( dZ, axis=1, keepdims=True )

        return np.dot( np.transpose( self.weights ) , dZ ) 
    
    def updateWeights(self, step):
        self.weights = -step*self.dW
        self.bias = -step*self.db

class FirstLayer(Layer):
    def __init__(self, trainData, testData):
        self.trainData = trainData
        self.testData = testData
    
    def getACacheValue(self):
        return self.trainData
    
    def forwardPropTrain(self):
        return self.trainData
    
    def forwardPropPredict(self):
        return self.testData

class LastLayer(Layer):
    #this layer must have sigmoid function as activation, because we use cross entropy cost function
    def __init__(self, weights, bias, activation, activationDerivative, trainResult):
        Layer.__init__(self, weights, bias, activation, activationDerivative)
        self.trainResult = trainResult

    def backwardProp(self,m):
        dZ = self.ACacheValue - self.trainResult
        self.dW = (1/m)*np.dot( dZ , self.previousLayer.getACacheValue().T )
        self.db = (1/m)*np.sum( dZ, axis=1, keepdims=True )
        return np.dot( np.transpose( self.weights ) , dZ )
    
    def getCost(self,m):
        return  -(1/m)* np.sum( 
            np.multiply( self.trainResult, np.log( self.ACacheValue ) ) + 
            np.multiply( 1 - self.trainResult,  np.log( 1 - self.ACacheValue ) )
            )