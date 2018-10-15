#! /usr/bin/python3
# -*- coding:utf-8 -*-

from multiprocessing import Pool
from network import *
from outils import *
from neurone import *
import xor
import os

# IMPORTANT
# Forcing numpy to be monothread to avoid over-threading
# DO -> export OPENBLAS_NUM_THREADS=1 before launching the script

training_image = [[0,0], [1,0], [0,1], [1,1]]
training_label = [[0],[1],[1],[0]]

# Number of processors
processors = 70

# Number of runs
runs =140 

# Number of iterations
step = 1
iterations = create_range(1,2000,step)

# Boolean : whether or not to change the learning_rate
learning_rates_tab = create_range(0.03,0.08,0.01) # list of learning rates
do3Dgraph = False


def doUrStuff(run, learning_rate) :
    global iterations
    global step
    global training_image
    global training_label

    error = [0] * len(iterations)
    layer1 = Layer(2,2, regulated_tanh, regulated_tanh_prim, learning_rate)
    layer2 = Layer(2,1, regulated_tanh, regulated_tanh_prim, learning_rate)
    network = Network([layer1,layer2], False)

    for i in range(len(iterations)) :
        error[i] += xor.main(step, iterations[i], network, training_image, training_label, run, True, learning_rate)
    return error

def map_XOR() :
    layer1 = Layer(2,2, regulated_tanh, regulated_tanh_prim, 0.05)
    layer2 = Layer(2,1, regulated_tanh, regulated_tanh_prim, 0.05)
    network = Network([layer1,layer2], False)
    for i in iterations :
        xor.main(step, i, network, training_image, training_label, 1, False)
    xor.map_XOR(network)


def multicoreGraph() :
    pool = Pool(processes = processors)
    # Global, not specific to a certain run

    errors = []
    for j in learning_rates_tab :
        error = [0] * len(iterations)
        results = pool.starmap(doUrStuff, tuple([(i, j) for i in range(1,runs+1)]))
        for result in results :
            for i in range(len(result)) :
                # We make the sum on all runs
                error[i] += result[i]
        for i in range(len(error)) :
            # We want an average with the number of runs but only made the sum by now
            error[i] /= runs
        errors.append(error)

    if not do3Dgraph :
        xor.graph2D(errors, iterations, learning_rates_tab, runs)
    else :
        xor.graph3D(errors, iterations, learning_rates_tab, runs, True)

multicoreGraph()
