#! /usr/bin/python3
# -*- coding:utf-8 -*-

from multiprocessing import Pool
from network import *
from outils import *
from neurone import *
import xor

training_image = [[0,0], [1,0], [0,1], [1,1]]
training_label = [[0],[1],[1],[0]]

# Number of processors
processors = 4

# Number of runs
runs = 8

# Number of iterations
step = 1000
iterations = create_range(1000,10000,step)

def doUrStuff(run) :
    global iterations
    global step
    global training_image
    global training_label

    error = [0] * len(iterations)
    layer1 = Layer(2,2,sigmoid, sigmoid_prim, 0.05)
    layer2 = Layer(2,1, sigmoid, sigmoid_prim, 0.05)
    network = Network([layer1,layer2], False)

    for i in range(len(iterations)) :
        error[i] += xor.main(step, iterations[i], network, training_image, training_label, run)
    return error

pool = Pool(processes = processors)

results = pool.map(doUrStuff, tuple([i for i in range(1,runs+1)]))

# Global, not specific to a certain run
error = [0] * len(iterations)

for result in results :
    for i in range(len(result)) :
        # We make the sum on all runs
        error[i] += result[i]

for i in range(len(error)) :
    # We want an average with the number of runs but only made the sum by now
    error[i] /= runs

print(error)
xor.great_3Dgraph(error, )
