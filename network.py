#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np


class Network:

    def __init__(self, layers, normalisation, optimiser=None):
        self.layers = layers
        self.do_normalisation = normalisation  # true or false
        self.optimiser = optimiser

    def normalisation(self, Input):
        # N = np.amax(Input)
        N = 255
        if N == 0:
            return Input
        else:
            return Input / N

    def feed_forward(self, Input):
        self.layers[0].compute(Input)
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].compute(self.layers[i].output)
        return self.layers[-1].output

    def compute_errors(self, expected, output):
        """returns the error for each neurone layer in a list from last tp first"""
        errors = []
        pre_error = -2 * (expected - output)
        for layer in reversed(self.layers):
            errors.append(pre_error * layer.activation.df(layer.activation_level))
            pre_error = layer.weights.transpose().dot(errors[-1])
        return errors[::-1]

    def backpropagation(self, Input, expected_output):
        errors = self.compute_errors(expected_output, self.layers[-1].output)

        delta_weights = []
        delta_weights.append(-self.layers[0].learning_rate *
                             np.outer(errors[0], Input))
        for i in range(1, len(self.layers)):
            delta_weights.append(-self.layers[i].learning_rate *
                                 np.outer(errors[i], self.layers[i-1].output))

        delta_bias = []
        for i in range(len(self.layers)):
            delta_bias.append(-self.layers[i].learning_rate * errors[i])

        for i in range(len(self.layers)):
            self.layers[i].update(delta_weights[i], delta_bias[i])

    def learning(self, Input, expected_output):
        Input = np.array(Input)
        expected_output = np.array(expected_output)
        if self.do_normalisation:
            Input = self.normalisation(Input)
        self.feed_forward(Input)
        self.backpropagation(Input, expected_output)

    def test(self, Input):
        Input = np.array(Input)
        if self.do_normalisation:
            Input = self.normalisation(Input)
        self.feed_forward(Input)
        return self.layers[-1].output
