#! /usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import math

hardlim = lambda x : int(x >= 0)

sigmoid = lambda x : 1 / (1 + math.exp(-x))
sigmoid_prim = lambda x : sigmoid(x) * (1 - sigmoid(x))

linear = lambda x : x
linear_prim = lambda x : 1
