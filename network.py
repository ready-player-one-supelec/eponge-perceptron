#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from outils import *
from neurone import *

class Network :

    def __init__(self, layers) :
        self.layers = layers

    def feed_forward(self, Input) :
        vec = np.array(Input)
        self.layers[0].compute(vec)
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
        sensibilities.reverse()
        return sensibilities

    def backpropagation(self, expected_output) :
        expected_output = np.array(expected_output)
        error = expected_output - self.layers[-1].output
        sensibilities = self.compute_sensibilities(error)




layer1 = Layer(2,2,sigmoid,sigmoid_prim)
layer2 = Layer(2,1,sigmoid,sigmoid_prim)
network = Network([layer1, layer2])
network.feed_forward([2,1])
network.backpropagation([1])
