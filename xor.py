#! /usr/bin/python3
# -*- coding:utf-8 -*-

from network import *
from neurone import *
from outils import *
import random
import matplotlib.pyplot as plt
import save
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

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
    plt.imshow(values, extent=[0,1,1,0], cmap='gray')
    plt.show()

banque_test= [([0,0], [0]), ([1,0], [1]), ([0,1], [1]), ([1,1],[0])]

def f(number_of_runs, learning_rate) :

    layer1 = Layer(2,2,sigmoid,sigmoid_prim, learning_rate)
    layer2 = Layer(2,1,sigmoid,sigmoid_prim, learning_rate)

    network = Network([layer1, layer2], normalisation = False)

    T = [0,1,2,3]
    for i in range(number_of_runs) :
        print(number_of_runs, i)
        random.shuffle(T)
        for j in T :
            network.learning(banque_test[T[j]][0],banque_test[T[j]][1])
    return (1- network.test([1,0]))**2 + network.test([1,1])**2 + network.test([0,0])**2 + (1-network.test([0,1]))**2


def great_3Dgraph() :
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    learning_rate = create_range(0.01,0.1,0.01)
    number_of_runs = create_range(1000,60000,1000)
    Z = np.zeros((len(number_of_runs), len(learning_rate)))
    for b in range(len(number_of_runs)) :
        for a in range(len(learning_rate)) :
            for i in range(10) :
                Z[b,a] += np.sqrt(f(number_of_runs[b], learning_rate[a]))
            Z[b,a] /= 10
    learning_rate, number_of_runs = np.meshgrid(learning_rate, number_of_runs)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Number of runs")
    ax.set_zlabel("Euclidian norm of the error (average with 10 repetitions)")
    ax.set_title("3D graph of the error for a XOR neural network")
    surf = ax.plot_surface(learning_rate, number_of_runs, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(20, 35) # 20 is the elevation parameter, 35 is azimut
    plt.savefig("data/Graph/XOR3D.png", transparent=False)
    plt.savefig("data/Graph/XOR3D_trans.png", transparent=True)
    plt.show()

great_3Dgraph()
