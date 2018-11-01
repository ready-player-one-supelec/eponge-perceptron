#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np


class Layer:

    def __init__(self, input_size, output_size, activation):
        # input_size is the number of neurones of the previous layer
        # output_size is the number of neurones of the current layer
        # activation is the activation function of this layer

        sigma = 1 / np.sqrt(input_size)
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.normal(0, sigma, size=[output_size, input_size])
        self.bias = np.random.normal(0, sigma, size=[output_size])
        self.activation = activation
        self.activation_level = np.zeros(output_size)
        self.output = np.zeros(output_size)

    def compute(self, Input):
        self.activation_level = np.dot(self.weights, Input) + self.bias
        self.output = self.activation.f(self.activation_level)

    def __len__(self):
        return len(self.output)

    def update(self, delta_weights, delta_bias):
        self.weights += delta_weights
        self.bias += delta_bias

    def add_to_weights(self, value):
        self.weights += value

    def add_to_bias(self, value):
        self.bias += value

    def set_weights(self, weights, bias):
        for i in range(len(weights)):
            for j in range(len(weights[i])):
                self.weights[i, j] = weights[i, j]
        for i in range(len(bias)):
            self.bias[i] = bias[i]

    def __str__(self):
        return f'{self.output_size}-{self.activation}'
