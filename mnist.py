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


def learning(network, training_image, training_label, times):
    liste = list(range(len(training_image)))
    for j in range(times) :
        random.shuffle(liste)
        for i,e in enumerate(liste) :
            if i % 100 == 0 :
                print("Learning #{} -".format(times),"Times :", j, " - Data : ", i)
            t = [0] * 10
            t[int(training_label[e])] = 1
            network.learning(training_image[e].flatten(),t)


def test_sample(network, test_image, test_label,times) :
    failure = 0
    success = 0

    for i in range(len(test_image)) :
        output = network.test(test_image[i].flatten())
        output = list(output)
        likely_number = output.index(max(output))
        if i % 100 == 0 :
            print("Testing #{} - success :".format(times), success,"- failures :", failure)
        if likely_number == int(test_label[i]) :
            success += 1
        else :
            failure += 1
    return success, failure


def create_network() :
    layer1 = Layer(784, 100, sigmoid, sigmoid_prim, 0.005)
    layer2 = Layer(100, 50, sigmoid, sigmoid_prim, 0.005)
    layer3 = Layer(50,10, sigmoid, sigmoid_prim, 0.005)
    network = Network([layer1,layer2,layer3])
    return network


def network_testing(times, training_image, training_label, test_image, test_label) :
    network = create_network()
    learning(network, training_image, training_label, times)
    training_success, training_failure = test_sample(network, training_image, training_label, times)
    test_success, test_failure = test_sample(network, test_image, test_label, times)
    test_failure_rate = test_failure / (test_success + test_failure)
    training_failure_rate = training_failure / (training_success + training_failure)
    save.save_network(network, "data/Networks Saved/best_network_times{}".format(times))
    return test_failure_rate, training_failure_rate


def results(times, training_failure_rate, test_failure_rate) :
    plt.plot(times,training_failure_rate,'r-', label="Taux d'échec sur l'échantillon d'apprentissage")
    plt.plot(times, test_failure_rate, 'b-', label="Taux d'échec sur l'échantillon test")
    plt.legend()
    plt.xlabel("Nombre de passages de l'échantillon d'apprentissage")
    plt.ylabel("Taux d'erreur")
    plt.ylim(0,1)
    plt.title("Erreurs du réseau sur les échantillons de test et d'apprentissage")
    plt.savefig("data/Graph/MNIST_trans.png", transparent= True)
    plt.savefig("data/Graph/MNIST.png", transparent=False)


#########################
# MAIN
#########################

def main(times, training_image, training_label, test_image, test_label) :
    test_failure_rate, training_failure_rate = network_testing(times, training_image, training_label, test_image, test_label)
    return test_failure_rate, training_failure_rate
