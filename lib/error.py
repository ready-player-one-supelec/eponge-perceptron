import numpy  as np

class Error:
    def error(self, output, output_before_activation, expected, last_layer):
        raise NotImplementedError()

class QuadratiError(Error):

    def error(self, output, output_before_activation, expected, last_layer):
        """returns a tuple containing e⁽N⁾ the last layer error and the loss""" 
        diff = (output - expected)**2
        return diff * last_layer.derivative(output_before_activation), np.sum(diff)

quadratic=QuadratiError()

