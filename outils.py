#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import math

hardlim = lambda x : int(x >= 0)

sigmoid = lambda x : 1 / (1 + math.exp(-x))
sigmoid_prim = lambda x : sigmoid(x) * (1 - sigmoid(x))

linear = lambda x : x
linear_prim = lambda x : 1

def tanh(x) :
    X = math.exp(2 * x)
    return (X - 1) / (X + 1)
tanh_prim = lambda x : 1 - tanh(x) ** 2

reLU = lambda x : x if x >= 0 else 0
reLU_prim = lambda x : 1 if x >=0 else 0
