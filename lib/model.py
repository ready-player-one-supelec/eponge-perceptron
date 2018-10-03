import numpy as np
import pickle
from itertools import chain

from .layer import Layer, InputLayer
from .error import quadratic, Error


class Model:

    def __init__(self, layers: list, error_calc=quadratic):
        self.layers: list[Layer] = layers
        self.error_calc: Error = error_calc
        self.learning_rate = 0.01

        self.weight_matrices = []
        self.values_before_activation = []
        self.values_after_activation = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def _initialize_vect(self):
        assert type(
            self.layers[0]) == InputLayer, 'you need in input as first layer'
        for layer in self.layers:
            self.values_before_activation.append(np.empty((layer.size, 1)))
            self.values_after_activation.append(
                np.empty((layer.size + layer.bias, 1)))
            self.values_after_activation[-1][-1, 0] = 1

    def initialize_random(self):
        self._initialize_vect()
        for precedent_layer, layer in zip(self.layers, self.layers[1:]):
            self.weight_matrices.append(
                np.random.randn(layer.size, precedent_layer.size + precedent_layer.bias) / precedent_layer.size)

    def initialize_from_weights(self, weights_list):
        assert len(weights_list) == (len(
            self.layers) - 1), "weigth list needs to be the same lenght as the number of layers"
        self._initialize_vect()
        self.weight_matrices = []
        for weights, precedent_layer, layer in zip(
            weights_list,
            self.layers,
            self.layers[1:]
        ):
            assert weights.shape == (
                layer.size, precedent_layer.size + precedent_layer.bias), "weight matrices are not the right size"
            self.weight_matrices.append(weights)

    def infer(self, input_vec):
        self.values_after_activation[0][:self.layers[0].size] = input_vec
        for layer, matrix, previous_value, before_act, after_act in zip(
            self.layers[1:],
            self.weight_matrices,
            self.values_after_activation,
            self.values_before_activation[1:],
            self.values_after_activation[1:]
        ):
            np.dot(matrix, previous_value, out=before_act)
            layer.activation.function(before_act, out=after_act[:layer.size])
        return self.values_after_activation[-1][:self.layers[-1].size]

    def save_weights(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.weight_matrices, file, protocol=0)

    def import_weights(self, filename):
        with open(filename, 'rb') as file:
            weights = pickle.load(file)
        self.initialize_from_weights(weights)

    def loss(self, infered, expected):
        return self.error_calc.loss(infered,expected)

    def backpropagate(self, infered, expected):
        error = self.error_calc.error(
            infered,
            self.values_before_activation[-1],
            expected,
            self.layers[-1]
        )
        for prev_layer, layer, after_act, before_act, weights in zip(
            reversed(self.layers[:-1]),
            reversed(self.layers),
            reversed(self.values_after_activation[:-1]),
            reversed(self.values_before_activation[:-1]),
            reversed(self.weight_matrices)
        ):
            update =  np.dot(error, np.transpose(after_act))
            out = np.empty_like(before_act[:prev_layer.size])
            prev_layer.activation.derivative(
                before_act[:prev_layer.size], out=out)
            error = np.dot(np.transpose(weights[:, :prev_layer.size]), error) * out
            yield update

    def update_weights(self, infered, expected):
        for weight, update in zip(reversed(self.weight_matrices), self.backpropagate(infered, expected)):
            np.subtract(weight, update * self.learning_rate, out=weight)
