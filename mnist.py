#! /usr/bin/python3
# -*- coding:utf-8 -*-

from network import *
from neurone import *
from outils import *
import numpy as np
import struct
import matplotlib.pyplot as plt
import random
import save

############################################################################
# MNIST
############################################################################

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape).astype(np.float_)


def learning(network, training_image, training_label, iteration, run):
    liste = list(range(len(training_image)))
    random.shuffle(liste)
    for i,e in enumerate(liste) :
        if i % 1000 == 0 :
            print("Run #{} -".format(run),"Iteration :", iteration, " - Data : ", i)
        t = [0] * 10
        t[int(training_label[e])] = 1
        network.learning(training_image[e].flatten(),t)


def test_sample(network, image, label,run) :
    failure = 0
    success = 0

    for i in range(len(image)) :
        output = network.test(image[i].flatten())
        output = list(output)
        likely_number = output.index(max(output))
        if i % 1000 == 0 :
            print("Testing #{} - success :".format(run), success,"- failures :", failure)
        if likely_number == int(label[i]) :
            success += 1
        else :
            failure += 1
    return success, failure


def create_network() :
    layer1 = Layer(784, 16, sigmoid, sigmoid_prim, 0.005)
    layer2 = Layer(16, 16, sigmoid, sigmoid_prim, 0.005)
    layer3 = Layer(16, 10, sigmoid, sigmoid_prim, 0.005)
    network = Network([layer1,layer2,layer3], normalisation = True)
    return network


def results(iterations, training_failure_rate, test_failure_rate, runs) :
    plt.plot(iterations, training_failure_rate,'r-', label="Failure rate on the training sample")
    plt.plot(iterations, test_failure_rate, 'b-', label="Failure rate on the test sample")
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Average failure rate (on {} runs)".format(runs))
    plt.ylim(0,1)
    plt.title("Failure rates of the network on training and test samples ({} runs)".format(runs))
    plt.savefig("data/Graph/MNIST_trans.png", transparent= True)
    plt.savefig("data/Graph/MNIST.png", transparent=False)


#########################
# MAIN
#########################

def main(training_image, training_label, test_image, test_label, network, iteration, run) :
    learning(network, training_image, training_label, iteration, run)
    training_success, training_failure = test_sample(network, training_image, training_label, run)
    test_success, test_failure = test_sample(network, test_image, test_label, run)
    test_failure_rate = test_failure / (test_success + test_failure)
    training_failure_rate = training_failure / (training_success + training_failure)
    return test_failure_rate, training_failure_rate
