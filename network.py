#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from optimiser import SGD


class Network:

    def __init__(self, layers, normalisation, optimiser=SGD(0.035)):
        self.layers = layers
        self.do_normalisation = normalisation  # true or false
        optimiser.initialize(self)
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
            errors.append(
                pre_error * layer.activation.df(layer.activation_level))
            pre_error = layer.weights.transpose().dot(errors[-1])
        return errors[::-1]

    def backpropagation(self, Input, expected_output):
        errors = self.compute_errors(expected_output, self.layers[-1].output)

        grad_weights = []
        grad_weights.append(np.outer(errors[0], Input))
        for i in range(1, len(self.layers)):
            grad_weights.append(np.outer(errors[i], self.layers[i-1].output))

        grad_biases = []
        for i in range(len(self.layers)):
            grad_biases.append(errors[i])
        self.optimiser.update_weight(grad_weights, grad_biases)

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
