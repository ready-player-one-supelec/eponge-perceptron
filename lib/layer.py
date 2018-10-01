
class Layer:
    def __init__(self, size, activation, bias=False):
        self.size = size
        self.activation = activation
        self.bias = bias
