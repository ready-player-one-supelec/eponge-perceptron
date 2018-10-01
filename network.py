#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from outils import *
from neurone import *

class Network :

    def __init__(self, layers, learning_rate) :
        self.layers = layers
        self.learning_rate = learning_rate

    def feed_forward(self, Input) :
        self.layers[0].compute(Input)
        for i in range(len(self.layers) - 1) :
            self.layers[i + 1].compute(self.layers[i].output)

    def compute_F_prim(self, layer_number) :
        layer = self.layers[layer_number]
        o = len(layer)
        F_prim = np.zeros((o, o))
        for i in range(len(F_prim)) :
            F_prim[i,i] = layer.f_prim(layer.activation_level[i])
        return F_prim

    def compute_sensibilities(self,error) :
        sensibilities = []
        F_prim = self.compute_F_prim(-1)

        sensibilities.append(-2 * F_prim.dot(error))
        for k in range(len(self.layers) - 2, -1, -1) :
            F_prim = self.compute_F_prim(k)
            sensibilities.append(F_prim.dot(self.layers[k+1].weights.transpose()).dot(sensibilities[-1]))
        return sensibilities[::-1]

    def backpropagation(self, Input, expected_output) :
        expected_output = np.array(expected_output)
        error = expected_output - self.layers[-1].output
        sensibilities = self.compute_sensibilities(error)
        print(type(np.array(sensibilities[0])))
        delta_weights = []
        delta_weights.append(-self.learning_rate * np.outer(sensibilities[0],Input))
        for i in range(1,len(self.layers)) :
            delta_weights.append(-self.learning_rate * np.outer(sensibilities[i],self.layers[i-1].output))

        delta_bias = []
        for i in range(len(self.layers)) :
            delta_bias.append(self.learning_rate * sensibilities[i])

        for i in range(len(self.layers)) :
            self.layers[i].update(delta_weights[i], delta_bias[i])

    def learning(self, Input, expected_output) :
        Input = np.array(Input)
        expected_output = np.array(expected_output)
        self.feed_forward(Input)
        self.backpropagation(Input, expected_output)



layer1 = Layer(2,2,sigmoid,sigmoid_prim)
layer2 = Layer(2,1,sigmoid,sigmoid_prim)
network = Network([layer1, layer2], 0.5)
network.learning([2,1], [1])
