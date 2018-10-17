#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from outils import *

class Layer :

    def __init__(self, entry, neurones, activation, diff_activation, learning_rate) :
        # entry is the number of neurones of the previous layer
        # neurones is the number of neurones of the current layer
        # activation is the activation function of this layer

        sigma = 1 / np.sqrt(entry)

        self.weights = np.random.normal(0, sigma, size=[neurones,entry])
        self.bias = np.random.normal(0, sigma, size=[neurones])
        self.f = activation
        self.f_prim = diff_activation
        self.activation_level = np.zeros(neurones)
        self.output = np.zeros(neurones)
        self.learning_rate = learning_rate

    def compute(self, Input) :
        self.activation_level = np.dot(self.weights, Input) - self.bias
        self.output=self.f(self.activation_level)

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
