import numpy as np

class Model :
    self.weight_matrices = []
    self.values_before_activation = []
    self.values_after_activation = []

    def __init__(self, input_size, layers):
        self.layers : list = layers
        self.input_size = input_size

    def add_layer(self, layer):
        self.layers.append(layer)

    def initialize_random(self): 
        // not DRY ¯\_(ツ)_/¯
        weight_matrices.append(np.random.randn(self.layers[0].size, self.input_size))
        self.values_before_activation.append(np.empty((self.layers[0].size,1)))
        self.values_after_activation.append(np.empty((self.layers[0].size,1)))
        for precedent_layer, layer in zip(self.layers, self.layers[1:]):
            self.weight_matrices.append(np.random.randn(layer.size,precedent_layer.size))
            self.values_before_activation.append(np.empty((layer.size,1)))
            self.values_after_activation.append(np.empty((layer[0].size,1)))
    
    def infer(self, input_vec):
        // not DRY ¯\_(ツ)_/¯
        np.dot(self.weight_matrices[0], input_vec, out=self.values_before_activation[0])
        layers[0].activation_function(self.values_before_activation[0], out=self.values_after_activation[0])

        for layer, matrix, previous_value, before_act, after_act in zip(
                self.layers[1:],
                self.weight_matrices[1:],
                self.values_after_activation, 
                self.values_before_activation[1:],
                self.values_after_activation[1:]
                ):
            np.dot(matrix, previous_value, out=before_act)
            layer.activation_function(before_act, out=after_act)
        return self.values_after_activation[-1]
