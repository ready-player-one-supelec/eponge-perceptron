#! /usr/bin/python3
# -*- coding:utf-8 -*-

import dill as pickle
from network import *
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
