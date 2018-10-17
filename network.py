#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from outils import *
import random
from neurone import *

class Network :

    def __init__(self, layers, normalisation) :
        self.layers = layers
        self.do_normalisation = normalisation # true or false

    def normalisation(self, Input) :
        # N = np.amax(Input)
        N = 255
        if N == 0 :
            return Input
        else :
            return Input / N

    def feed_forward(self, Input) :
        self.layers[0].compute(Input)
        for i in range(len(self.layers) - 1) :
            self.layers[i + 1].compute(self.layers[i].output)

    def compute_F_prim(self, layer_number) :
        layer = self.layers[layer_number]
        F_prim = np.diag(layer.f_prim(layer.activation_level))
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
        error = expected_output - self.layers[-1].output
        sensibilities = self.compute_sensibilities(error)

        delta_weights = []
        delta_weights.append(-self.layers[0].learning_rate * np.outer(sensibilities[0],Input))
        for i in range(1,len(self.layers)) :
            delta_weights.append(-self.layers[i].learning_rate * np.outer(sensibilities[i],self.layers[i-1].output))

        delta_bias = []
        for i in range(len(self.layers)) :
            delta_bias.append(self.layers[i].learning_rate * sensibilities[i])

        for i in range(len(self.layers)) :
            self.layers[i].update(delta_weights[i], delta_bias[i])

    def learning(self, Input, expected_output) :
        Input = np.array(Input)
        expected_output = np.array(expected_output)
        if self.do_normalisation :
            Input = self.normalisation(Input)
        self.feed_forward(Input)
        self.backpropagation(Input, expected_output)

    def test(self, Input) :
        Input = np.array(Input)
        if self.do_normalisation :
            Input = self.normalisation(Input)
        self.feed_forward(Input)
        return self.layers[-1].output
