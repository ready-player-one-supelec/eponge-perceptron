#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from outils import *
from neurone import *

class Network :

    def __init__(self, layers) :
        self.layers = layers

    def feed_forward(self, Input) :
        vec = np.array(Input)
        self.layers[0].compute(vec)
        for i in range(len(self.layers) - 1) :
            self.layers[i + 1].compute(self.layers[i].output)
