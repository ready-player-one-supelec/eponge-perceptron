import lib
import numpy as np
import unittest

from math import tanh, exp

def d_tanh(x):
    return 1 - tanh(x)**2

def sigmoid(x):
    return 1 / (1 + exp(-x))
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class TestActivation(unittest.TestCase):

    def setUp(self):
        self.identity = lib.activation.Identity()
        self.tanh = lib.activation.Tanh()
        self.sigmoid = lib.activation.Sigmoid()
        self.vec_in = np.array([[1],[2],[3]])
        self.vec_out = np.empty((3,1))

    def test_identity(self):
        #derivative
        self.identity.function(self.vec_in, self.vec_out)
        self.assertTrue(np.all(self.vec_in == self.vec_out))
        #function
        self.identity.derivative(self.vec_in, self.vec_out)
        self.assertTrue(np.all(self.vec_out == np.ones_like(self.vec_out)))

    def test_tanh(self):
        #function
        self.tanh.function(self.vec_in, self.vec_out)
        self.assertTrue(np.all(self.vec_out == np.array([[tanh(1)],[tanh(2)],[tanh(3)]])))
        #derivative
        self.tanh.derivative(self.vec_in, self.vec_out)
        self.assertTrue(np.all(self.vec_out == np.array([[d_tanh(1)],[d_tanh(2)],[d_tanh(3)]])))

    def test_sigmoid(self):
        #function
        self.sigmoid.function(self.vec_in, self.vec_out)
        self.assertTrue(np.all(self.vec_out == np.array([[sigmoid(1)],[sigmoid(2)],[sigmoid(3)]])))
        #derivative
        self.sigmoid.derivative(self.vec_in, self.vec_out)
        self.assertTrue(np.all(self.vec_out == np.array([[d_sigmoid(1)],[d_sigmoid(2)],[d_sigmoid(3)]])))


class TestModel1(unittest.TestCase):
    def setUp(self):
        self.model = lib.model.Model([
            lib.layer.InputLayer(1),
            lib.layer.Layer(1, lib.activation.identity)
        ])
        self.model.initialize_from_weights([np.array([[1]], dtype=np.float_)])
        
    def testForward(self):
        input_vec = np.array([[5]],dtype=np.float_)
        output_vec = self.model.infer(input_vec)
        self.assertTrue(np.all(output_vec == input_vec))

    def testBackward(self):
        input_vec = np.array([[5]],dtype=np.float_)
        output_vec = self.model.infer(input_vec)
        expected =  np.array([[4]],dtype=np.float_)
        for update in self.model.backpropagate(output_vec, expected):
            print(update)

class TestModel2(unittest.TestCase):
    def setUp(self):
        self.model = lib.model.Model([
            lib.layer.InputLayer(1),
            lib.layer.Layer(1, lib.activation.tanh),
            lib.layer.Layer(1, lib.activation.tanh)
        ])
        self.model.initialize_from_weights([np.array([[1]], dtype=np.float_), np.array([[1]], dtype=np.float_)])
    
    def testForward(self):
        input_vec = np.array([[5]],dtype=np.float_)
        output_vec = self.model.infer(input_vec)
        self.assertTrue(np.all(output_vec == np.array([[tanh(tanh(5))]],dtype=np.float_)))

class TestModel3(unittest.TestCase):
    def setUp(self):
        self.model = lib.model.Model([
            lib.layer.InputLayer(2),
            lib.layer.Layer(2, lib.activation.identity),
            lib.layer.Layer(1, lib.activation.identity)
        ])
        self.model.initialize_from_weights([
            np.array([[1,2],[3,4]], dtype=np.float_),
            np.array([[1,1]], dtype=np.float_)
            ])
    
    def testForward(self):
        input_vec = np.array([[1],[2]],dtype=np.float_)
        output_vec = self.model.infer(input_vec)
        self.assertTrue(np.all(output_vec == np.array([[16]],dtype=np.float_)))


if __name__ == '__main__':
    unittest.main()
