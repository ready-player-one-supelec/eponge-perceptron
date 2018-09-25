import lib
import numpy as np
import unittest

from math import tanh

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
        vec_in = np.random.random((10, 1))
        vec_out = np.empty((10, 1))
        self.identity.derivative(vec_in, vec_out)
        self.assertTrue(np.all(vec_out == np.ones_like(vec_out)))


if __name__ == '__main__':
    unittest.main()
