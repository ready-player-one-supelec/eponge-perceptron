#! /usr/bin/python3
# -*- coding:utf-8 -*-

from network import Network
from neurone import Layer
import activation
import numpy as np
import struct

############################################################################
# MNIST
############################################################################


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape).astype(np.float_)


def learning(step, liste, network, training_image, training_label, iteration, run):
    print("Run #{} -".format(run), "Iteration :",
          iteration // 60000, "Data : ", iteration % 60000)
    for i, e in enumerate(liste[iteration % 60000: (iteration + step - 1) % 60000 + 1]):
        t = [0] * 10
        t[int(training_label[e])] = 1
        network.learning(training_image[e].flatten(), t)


def test_sample(network, image, label, run):
    failure = 0
    success = 0

    for i in range(len(image)):
        output = network.test(image[i].flatten())
        output = list(output)
        likely_number = output.index(max(output))
        if i % 10000 == 0:
            print("Testing #{} - success :".format(run),
                  success, "- failures :", failure)
        if likely_number == int(label[i]):
            success += 1
        else:
            failure += 1
    return success, failure


def create_network(learning_rate):
    layer1 = Layer(784, 16, activation.regulated_tanh, learning_rate)
    layer2 = Layer(16, 16, activation.regulated_tanh, learning_rate)
    layer3 = Layer(16, 10, activation.regulated_tanh, learning_rate)
    network = Network([layer1, layer2, layer3], normalisation=True)
    return network


#########################
# MAIN
#########################

def main(step, liste, training_image, training_label, test_image, test_label, network, iteration, run):
    learning(step, liste, network, training_image,
             training_label, iteration, run)
    training_success, training_failure = test_sample(
        network, training_image, training_label, run)
    test_success, test_failure = test_sample(
        network, test_image, test_label, run)
    test_failure_rate = test_failure / (test_success + test_failure)
    training_failure_rate = training_failure / \
        (training_success + training_failure)
    return test_failure_rate, training_failure_rate
