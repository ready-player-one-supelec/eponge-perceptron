import numpy as np
import pickle


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
        # not DRY ¯\_(ツ)_/¯
        self._initialize_vect()
        weight_matrices.append(np.random.randn(
            self.layers[0].size, self.input_size))
        for precedent_layer, layer in zip(self.layers, self.layers[1:]):
            self.weight_matrices.append(
                np.random.randn(layer.size, precedent_layer.size))

    def initialize_from_weights(self, weights_list):
        assert len(weights_list) == len(
            self.layers), "weigth list needs to be the same lenght as the number of layers"
        assert weights_list[0].shape == (
            self.layers[0].size, self.input_size), "weight matrices are not the right size"
        self._initialize_vect()
        self.weight_matrices = []
        self.weight_matrices.append(weights_list[0])
        for weights, precedent_layer, layer in zip(weights_list[1:], self.layers, self.layers[1:]):
            assert weights.shape == (
                layer.size, precedent_layer.size), "weight matrices are not the right size"
            self.weight_matrices.append(weights)

    def infer(self, input_vec):
        # not DRY ¯\_(ツ)_/¯
        np.dot(self.weight_matrices[0], input_vec,
               out=self.values_before_activation[0])
        self.layers[0].activation.function(
            self.values_before_activation[0], out=self.values_after_activation[0])

        for layer, matrix, previous_value, before_act, after_act in zip(
                self.layers[1:],
                self.weight_matrices[1:],
                self.values_after_activation,
                self.values_before_activation[1:],
                self.values_after_activation[1:]
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
