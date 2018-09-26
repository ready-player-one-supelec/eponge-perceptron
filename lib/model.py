import numpy as np
import pickle
from itertools import chain

from .layer import Layer

class Model:

    def __init__(self, input_size : int, layers : list):
        self.layers = layers
        self.input_size = input_size

        self.weight_matrices = []
        self.values_before_activation = []
        self.values_after_activation = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def _initialize_vect(self):
        for layer in self.layers:
            self.values_before_activation.append(np.empty((layer.size, 1)))
            self.values_after_activation.append(np.empty((layer.size, 1)))

    def initialize_random(self):
        self._initialize_vect()
        weight_matrices.append(np.random.randn(
            self.layers[0].size, self.input_size))
        for precedent_layer, layer in zip(self.layers, self.layers[1:]):
            self.weight_matrices.append(
                np.random.randn(layer.size, precedent_layer.size))

    def initialize_from_weights(self, weights_list):
        assert len(weights_list) == len(
            self.layers), "weigth list needs to be the same lenght as the number of layers"
        self._initialize_vect()
        self.weight_matrices = []
        for weights, precedent_layer, layer in zip(
                weights_list, 
                chain([Layer(self.input_size, None)], self.layers), 
                self.layers
            ):
            assert weights.shape == (
                layer.size, precedent_layer.size), "weight matrices are not the right size"
            self.weight_matrices.append(weights)

    def infer(self, input_vec):
        for layer, matrix, previous_value, before_act, after_act in zip(
                self.layers,
                self.weight_matrices,
                chain([input_vec], self.values_after_activation),
                self.values_before_activation,
                self.values_after_activation
            ):
            np.dot(matrix, previous_value, out=before_act)
            layer.activation.function(before_act, out=after_act)
        return self.values_after_activation[-1]

    def save_weights(self, filename):
        with open(filename, 'wb') as file :
            pickle.dump(self.weight_matrices, file, protocol=0)

    def import_weights(self, filename):
        with open(filename, 'rb') as file :
            weights = pickle.load(file)
        self.initialize_from_weights(weights)
