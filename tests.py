import lib
import numpy as np
import unittest

from math import tanh

def d_tanh(x):
    return 1 - tanh(x)**2

class TestActivation(unittest.TestCase):

    def setUp(self):
        self.identity = lib.activation.identity
        self.tanh = lib.activation.tanh
        self.vec_in = np.array([[1],[2],[3]])
        self.vec_out = np.empty((3,1))


    def test_identity(self):
        self.identity.function(self.vec_in, self.vec_out)
        self.assertTrue(np.all(self.vec_in == self.vec_out))

    def test_identity_derivative(self):
        self.identity.derivative(self.vec_in, self.vec_out)
        self.assertTrue(np.all(self.vec_out == np.ones_like(self.vec_out)))

    def test_tanh(self):
        self.tanh.function(self.vec_in, self.vec_out)
        self.assertTrue(np.all(self.vec_out == np.array([[tanh(1)],[tanh(2)],[tanh(3)]])))

    def test_tanh_derivative(self):
        self.tanh.derivative(self.vec_in, self.vec_out)
        self.assertTrue(np.all(self.vec_out == np.array([[d_tanh(1)],[d_tanh(2)],[d_tanh(3)]])))


if __name__ == '__main__':
    unittest.main()
