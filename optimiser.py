import numpy as np
from abc import ABC, abstractmethod


class Optimiser(ABC):

    # in case we need to initialize things that cannot been done in __init__
    @abstractmethod
    def initialize(self, network): ...

    # method called by the betwork to update its weights
    @abstractmethod
    def update_weight(self, gradient_weights, gradient_biases): ...


class SGD(Optimiser):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_weight(self, gradient_weights, gradient_biases):
        for layer, grad_w, grad_b in zip(
            self.network.layers,
            gradient_weights,
            gradient_biases
        ):
            layer.add_to_weights((-self.learning_rate) * grad_w)
            layer.add_to_bias((-self.learning_rate) * grad_b)

    def initialize(self, network):
        self.network = network


class RMSprop(Optimiser):
    def __init__(self, learning_rate, decay, epsilon=10**(-8)):
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon

    def initialize(self, network):
        self.RMS_matrices = []
        self.RMS_biases = []
        self.network = network
        for layer in network.layers:
            self.RMS_matrices.append(np.zeros_like(layer.weights))
            self.RMS_biases.append(np.zeros_like(layer.bias))

    def update_weight(self, gradient_weights, gradient_biases):
        for i in range(len(self.network.layers)):
            RMS_matrix = self.RMS_matrices[i]
            RMS_bias = self.RMS_biases[i]

            # update the RMS moving average
            RMS_matrix = self.decay * RMS_matrix + \
                (1 - self.decay) * (gradient_weights[i]**2)
            RMS_bias = self.decay * RMS_bias + \
                (1 - self.decay) * (gradient_biases[i]**2)
            self.RMS_matrices[i] = RMS_matrix
            self.RMS_biases[i] = RMS_bias

            # compute the weight update
            update_matrix = self.learning_rate * \
                gradient_weights[i] / (RMS_matrix + self.epsilon)**(1/2)
            update_bias = self.learning_rate * \
                gradient_biases[i] / (RMS_bias + self.epsilon)**(1/2)

            # apply the update
            self.network.layers[i].add_to_weights(-update_matrix)
            self.network.layers[i].add_to_bias(-update_bias)
