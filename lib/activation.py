import numpy as np

class Activation:
    def function(self, vector, out):
        raise NotImplementedError()
    def derivative(self, vector, out):
        raise NotImplementedError()

class Identity(Activation):
    def function(self, vector, out):
        np.copyto(out, vector)

    def derivative(self, vector, out):
        out.fill(1)

class Tanh(Activation):
    def function(self, vector, out):
        np.tanh(vector, out=out)

    def derivative(self, vector, out):
        np.tanh(vector, out=out)
        out *= out
        np.subtract(out, 1, out=out)
        np.negative(out, out=out)

class Sigmoid(Activation):
    def function(self, vector, out):
        np.exp(-vector, out=out)
        out += 1
        np.divide(1,out,out=out)

    def derivative(self, vector, out):
        self.function(vector, out=out)
        np.multiply(out, 1 - out, out=out)

identity = Identity()
tanh = Tanh()
sigmoid = Sigmoid()
