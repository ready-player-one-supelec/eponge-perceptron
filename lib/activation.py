import numpy as np

class Activation:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative


def id_activation(vector, out):
    np.copyto(out, vector)

def id_derivative(vector, out):
    out.fill(1)


def tanh_activation(vector, out):
    np.tanh(vector, out=out)


def tanh_derivative(vector, out):
    np.tanh(vector, out=out)
    out *= out
    np.subtract(out, 1, out=out)
    np.negative(out, out=out)


identity = Activation(id_activation, id_derivative)
tanh = Activation(tanh_activation, tanh_derivative)
