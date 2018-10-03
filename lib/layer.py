from .activation import identity

class Layer:
    def __init__(self, size, activation, bias=False):
        self.size = size
        self.activation = activation
        self.bias = bias

class InputLayer(Layer):
    def __init__(self, size, bias=False):
        super().__init__(size, identity, bias=bias)