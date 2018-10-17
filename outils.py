#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import math

sigmoid = lambda x : 1 / (1 + np.exp(-x))
sigmoid_prim = lambda x : sigmoid(x) * (1 - sigmoid(x))

tanh=np.tanh
tanh_prim = lambda x : 1 - tanh(x) ** 2

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
