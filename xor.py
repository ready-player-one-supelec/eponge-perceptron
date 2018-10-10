#! /usr/bin/python3
# -*- coding:utf-8 -*-

from network import *
from neurone import *
from outils import *
import numpy as np
import matplotlib.pyplot as plt
import save

###########################################################################
# TEST DU XOR
###########################################################################

def graph(network) :
    N = 100
    values = [[0.0 for i in range(N+1)] for j in range(N+1)]
    for i in range(N+1) :
        for j in range(N+1) :
            tab = [i*1.0/ N,j*1.0 / N]
            values[i][j] = network.test(tab)[0]
    plt.imshow(values, extent=[0,1,0,1], cmap='gray')
    plt.show()

layer1 = Layer(2,4,sigmoid,sigmoid_prim)
layer2 = Layer(4,1,sigmoid,sigmoid_prim)

network = Network([layer1, layer2], 0.01)

banque_test= [([0,0], [0]), ([1,0], [1]), ([0,1], [1]), ([1,1],[0])]

def f() :
    for i in range(60000) :
        print(i)
        T = [0,1,2,3]
        random.shuffle(T)
        for j in T :
            network.learning(banque_test[T[j]][0],banque_test[T[j]][1])
    return (1- network.test([1,0]))**2 + network.test([1,1])**2 + network.test([0,0])**2 + (1-network.test([0,1]))**2

print(f())
print(network.test([0,0]))
print(network.test([1,0]))
print(network.test([0,1]))
print(network.test([1,1]))


graph(network)
