import numpy as np


class Error:
    def error(self, output, output_before_activation, expected, last_layer):
        raise NotImplementedError()
    def loss(self, output, expected):
        raise NotImplementedError()


class QuadratiError(Error):

    def error(self, output, output_before_activation, expected, last_layer):
        """returns a tuple containing e⁽N⁾ the last layer error and the loss"""
        diff = (output - expected)
        ret = np.empty_like(output_before_activation)
        last_layer.activation.derivative(output_before_activation, out=ret)
        return diff * ret

    def loss(self, output, expected):
        return np.sum((output - expected)**2)



quadratic = QuadratiError()
