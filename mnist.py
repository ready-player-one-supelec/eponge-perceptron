#! /usr/bin/python3
# -*- coding:utf-8 -*-

from network import *
from neurone import *
from outils import *
import numpy as np
import struct
import matplotlib.pyplot as plt
import save
import random
import matplotlib.pyplot as plt

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
            print("Times :", j, " - i : ", i)
            t = [0] * 10
            t[int(training_label[e])] = 1
            network.learning(training_image[e].flatten(),t)
            if i >= 10 :
                break

def test_sample(network, test_image, test_label) :
    failure = 0
    success = 0

    for i in range(len(test_image)) :
        if i >= 100 :
            break
        output = network.test(test_image[i].flatten())
        output = list(output)
        likely_number = output.index(max(output))
        print("output:",output)
        print("guess:",likely_number)
        print("truth:",int(test_label[i]))
        print("success:", success)
        print("failures:", failure)
        print("-------------------------")
        if likely_number == int(test_label[i]) :
            success += 1
        else :
            failure += 1
    return success, failure

def create_network() :
    layer1 = Layer(784, 100, sigmoid, sigmoid_prim)
    layer2 = Layer(100, 50, sigmoid, sigmoid_prim)
    layer3 = Layer(50,10, sigmoid, sigmoid_prim)
    network = Network([layer1,layer2,layer3], 0.005)
    return network

training_image = read_idx("train-images-idx3-ubyte")
training_label = read_idx("train-labels-idx1-ubyte")
test_image = read_idx("t10k-images-idx3-ubyte")
test_label = read_idx("t10k-labels-idx1-ubyte")

times = [i for i in range(10)]
test_failure_rate = []
training_failure_rate = []

for i in times :
    network = create_network()
    learning(network, training_image, training_label, i)
    training_success, training_failure = test_sample(network, training_image, training_label)
    test_success, test_failure = test_sample(network, test_image, test_label)
    test_failure_rate.append(test_failure / (test_success + test_failure))
    training_failure_rate.append(training_failure / (training_success + training_failure))

plt.plot(times,training_failure_rate,'r-', label="Taux d'échec sur l'échantillon d'apprentissage")
plt.plot(times, test_failure_rate, 'b-', label="Taux d'échec sur l'échantillon test")
plt.legend()
plt.xlabel("Nombre de passage de l'échantillon d'apprentissage")
plt.ylabel("Taux d'erreur")
plt.ylim(0,1)
plt.title("Erreurs du réseau sur les échantillons de test et d'apprentissage")
plt.show()
