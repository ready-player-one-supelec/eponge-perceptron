#! /usr/bin/python3
# -*- coding:utf-8 -*-

import dill as pickle
from network import *
from neurone import *

def save_network(network, filename) :

    layers = []

    for layer in network.layers :
        X = {"weights" : layer.weights, "bias" : layer.bias, "f" : layer.f, "f_prim" : layer.f_prim}
        layers.append(X)
    S = {"learning_rate" : network.learning_rate, "layers" : layers}
    pickle.dump(S, open(filename,"wb"))


def load_network(filename) :

    L = pickle.load(open(filename, "rb"))

    learning_rate = L["learning_rate"]
    layers = L["layers"]
    tab = []
    for layer in layers :
        neurones = len(layer["weights"])
        entry = len(layer["weights"][0])
        f = layer["f"]
        f_prim = layer["f_prim"]
        new_layer = Layer(entry,neurones,f, f_prim)
        new_layer.set_weights(layer["weights"], layer["bias"])
        tab.append(new_layer)
    network = Network(tab, learning_rate)
    return network
