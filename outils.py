#! /usr/bin/python3
# -*- coding:utf-8 -*-

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


def create_range(start,end,step) :
    # Can create range with non integer but float
    # Be careful, end is included unlike classical range functions
    arrondi = int(-math.log10(step))
    if arrondi < 1 :
        return [start + i * step for i in range(int((end-start) // step) + 1)]
    else :
        return [round(start+i*step, arrondi) for i in range(int((end-start) // step) + 1)]

regulated_tanh = lambda x : 1.7159 * tanh(2 * x / 3)
regulated_tanh_prim = lambda x : 1.7159 * (2 / 3) * tanh_prim(2 * x / 3)
