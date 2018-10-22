import numpy as np 
import math

class RMSProp():
    def __init__(self, gamma , epsilon):
        self.gamma = gamma
        self.epsilon = epsilon
        self.error = 0

    def sumAll(delta_X):
        res = 0
        for obj in delta_X:
            res += np.square(obj).sum()
        return res
    
    def updateError(self, delta_weights, delta_bias):
        # computes E(gradient)[n] = gamma * E(gradient)[n-1] + ( 1 - gamma ) * gradient ^2
        self.error = self.gamma * self.error + (1 - self.gamma) * (sumAll(delta_bias) + sumAll(delta_weights))
    
    def returnCoefficient(self):
        return 1/(math.sqrt( self.error + self.epsilon ))