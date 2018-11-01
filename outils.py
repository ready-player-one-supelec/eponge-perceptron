#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import math
import struct


def sigmoid(x): return 1 / (1 + np.exp(-x))


def sigmoid_prim(x): return sigmoid(x) * (1 - sigmoid(x))


tanh = np.tanh


def tanh_prim(x): return 1 - tanh(x) ** 2


def create_range(start, end, step):
    # Can create range with non integer but float
    # Be careful, end is included unlike classical range functions
    arrondi = int(-math.log10(step))
    if arrondi < 1:
        return [start + i * step for i in range(int((end-start) // step) + 1)]
    else:
        return [round(start+i*step, arrondi) for i in range(int((end-start) // step) + 1)]


def regulated_tanh(x): return 1.7159 * tanh(2 * x / 3)


def regulated_tanh_prim(x): return 1.7159 * (2 / 3) * tanh_prim(2 * x / 3)


def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape).astype(np.float_)
