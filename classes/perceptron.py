from classes import layer
import numpy as np 

class Perceptron():
    
    def __init__(self, listDimension, trainData, testData, activation, activationDerivative, trainResult):
        self.trainDataAmount = trainData.shape[1]
        self.listDimension = listDimension
        self.listLayers = [layer.FirstLayer(trainData, testData)]
        for i in range(1,len(listDimension)-1):
            self.listLayers.append( 
                layer.Layer(self.initWeights( (listDimension[i], listDimension[i-1]) ), 
                self.initBias(listDimension[i]),
                activation,
                activationDerivative
                )
            )
        self.listLayers.append(
            layer.LastLayer(self.initWeights( (listDimension[-1], listDimension[-2]) ), 
                self.initBias(listDimension[-1]),
                activation,
                activationDerivative,
                trainResult
            )
        )
        self.listLayers[0].setNextLayer( self.listLayers[1] )
        for i in range(1,len(self.listLayers)-1):
            self.listLayers[i].setPreviousLayer( self.listLayers[i-1] )
            self.listLayers[i].setNextLayer( self.listLayers[i+1] )
        self.listLayers[-1].setPreviousLayer(self.listLayers[-2])

    def initWeights(self, dimension):
        # return initialized weights and bias 
        return np.random.rand(dimension[0], dimension[1])
        
    def initBias(self, dimension): 
        return np.random.rand(dimension,1)
    
    def getPrediction(self):
        return self.listLayers[-1].forwardPropPredict()
    

    def iterateOnce(self,aff):
        m = self.trainDataAmount
        self.listLayers[-1].forwardPropTrain()
        if aff:
            print(self.listLayers[-1].getCost(m))
        self.listLayers[1].backwardProp(m)
        for i in range(1, len(self.listLayers)):
            self.listLayers[i].updateWeights(0.01)