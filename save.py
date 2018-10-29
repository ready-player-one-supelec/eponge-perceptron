#! /usr/bin/python3
# -*- coding:utf-8 -*-

import pickle as pickle
from network import *
import xor
from neurone import *

def save_network(network, filename) :
    layers = []
    for layer in network.layers :
        X = {"weights" : layer.weights, "bias" : layer.bias, "f" : layer.f, "f_prim" : layer.f_prim, "learning_rate" : layer.learning_rate}
        layers.append(X)
    pickle.dump(layers, open(filename,"wb"))


def load_network(filename) :

    layers = pickle.load(open(filename, "rb"))
    tab = []
    for layer in layers :
        neurones = len(layer["weights"])
        entry = len(layer["weights"][0])
        f = layer["f"]
        f_prim = layer["f_prim"]
        learning_rate = layer["learning_rate"]
        new_layer = Layer(entry,neurones,f, f_prim, learning_rate)
        new_layer.set_weights(layer["weights"], layer["bias"])
        tab.append(new_layer)
    network = Network(tab)
    return network

def save_data_for_3D_XOR(errors, iterations, learning_rates, runs, filename) :
    X = {"errors" : errors, "iterations" : iterations, "learning_rates" : learning_rates, "runs": runs}
    pickle.dump(X, open(filename, "wb"))

def load_data_for_3D_XOR(filename) :
    X = pickle.load(open(filename, "rb"))
    errors = X["errors"]
    iterations = X["iterations"]
    learning_rates = X["learning_rates"]
    runs = X["runs"]
    print(iterations)
    print(learning_rates)
    print(errors)
    xor.graph3D(errors, iterations, learning_rates, runs, False)
