#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from outils import *

class Layer :

    def __init__(self, entry, neurones, activation, diff_activation) :
        # entry is the number of neurones of the previous layer
        # neurones is the number of neurones of the current layer
        # activation is the activation function of this layer

        self.weights = 2 * np.random.rand(neurones, entry) - np.ones((neurones,entry))
        self.bias = 2 * np.random.rand(neurones) - np.ones(neurones)
        self.f = activation
        self.f_prim = diff_activation
        self.activation_level = np.zeros(neurones)
        self.output = np.zeros(neurones)

    def compute(self, Input) :
        self.activation_level = np.dot(self.weights, Input) - self.bias
        for i in range(len(self.output)) :
            self.output[i] = self.f(self.activation_level[i])

    def __len__(self) :
        return len(self.output)

    def update(self, delta_weights, delta_bias) :
        self.weights += delta_weights
        self.bias += delta_bias

    def set_weights(self, weights, bias) :
        for i in range(len(weights)) :
            for j in range(len(weights[i])) :
                self.weights[i,j] = weights[i,j]
        for i in range(len(bias)) :
            self.bias[i] = bias[i]
