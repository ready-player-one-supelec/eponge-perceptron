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


def map_XOR(network) :
    N = 100
    values = [[0.0 for i in range(N+1)] for j in range(N+1)]
    for i in range(N+1) :
        for j in range(N+1) :
            tab = [i*1.0/ N,j*1.0 / N]
            values[i][j] = network.test(tab)[0]
    plt.imshow(values, extent=[0,1,1,0], cmap='gray')
    plt.show()


def main(step, iteration, network, training_image, training_label, run, do_test) :
    liste = list(range(len(training_image)))
    random.shuffle(liste)
    print("Run #{} -".format(run),"Iteration :", iteration)
    for a in range(step) :
        for e in liste :
            network.learning(training_image[e],training_label[e])
    error = 0
    if do_test :
        for i in range(len(training_image)) :
            tmp = network.test(training_image[i]) - training_label[i]
            error += tmp[0] ** 2
    return np.sqrt(error)


def graph2D(error, iterations, learning_rate, runs) :
    # constant learning rate
    fig = plt.figure()
    plt.plot(iterations, error, 'b-', label="Learning rate = {}".format(learning_rate))
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Euclidian norm of the error (average with {} runs)".format(runs))
    plt.title("2D graph of the error for a XOR neural network for a constant learning rate")
    plt.savefig("data/Graph/XOR2D.png", transparent=False)
    plt.savefig("data/Graph/XOR2D_trans.png", transparent= True)


def graph3D(errors, iterations, learning_rates, runs, save_data) :
    # changing learning rates
    # save_data is a boolean : whether or not to save the errors to visualize later
    # It is useful to ignore it if we just loaded some data
    Z = np.transpose(np.array(errors))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(learning_rates, iterations)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Number of iterations")
    ax.set_zlabel("Euclidian norm of the error (average with {} runs)".format(runs))
    ax.set_title("3D graph of the error for a XOR neural network")
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    if save_data :
        save.save_data_for_3D_XOR(errors, iterations, learning_rates, runs, "data/Graph/graph3D_data")
