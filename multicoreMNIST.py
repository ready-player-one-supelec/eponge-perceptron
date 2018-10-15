#! /usr/bin/python3
# -*- coding:utf-8 -*-

from multiprocessing import Pool
import mnist
import numpy as np
import os
from outils import *
import random

# IMPORTANT
# Forcing numpy to be monothread to avoid over-threading
# DO -> export OPENBLAS_NUM_THREADS=1 before launching the script

# Number of processors
processors = 70

# Number of runs (for the average)
runs = 280

# Times we learn on the training sample
step = 10000 # we want to make a point even if the whole training set has not been coped with yet
number_of_complete_iterations = 50
iterations = create_range(step, number_of_complete_iterations * 60000 - 1, step)
# 60000 is the bulk of the training sample,
# so the first number is the actual number of iterations of the set


training_image = mnist.read_idx("data/MNIST/train-images-idx3-ubyte")
training_label = mnist.read_idx("data/MNIST/train-labels-idx1-ubyte")
test_image = mnist.read_idx("data/MNIST/t10k-images-idx3-ubyte")
test_label = mnist.read_idx("data/MNIST/t10k-labels-idx1-ubyte")


def doUrStuff(run) :
    global training_image
    global training_label
    global test_image
    global test_label
    global test_failure_rate
    global training_failure_rate
    global iterations
    network = mnist.create_network()
    test_failure_rate = [0] * len(iterations)
    training_failure_rate = [0] * len(iterations)
    liste = list(range(len(training_image)))
    random.shuffle(liste)
    for i in range(len(iterations)) :
        if iterations[i] % 60000 == 0 :
            random.shuffle(liste)
        a,b = mnist.main(step, liste, training_image, training_label, test_image, test_label, network, iterations[i], run)
        test_failure_rate[i] += a
        training_failure_rate[i] += b
    return test_failure_rate, training_failure_rate

pool = Pool(processes = processors)

# distributing the run to every core
times = [i for i in range(1, runs + 1)]
results = pool.map(doUrStuff, tuple(times))

# Cette fois-ci global et non spécifique à un run donné
test_failure_rate = [0] * len(iterations)
training_failure_rate = [0] * len(iterations)


for result in results :
    for i in range(len(result[0])) :
        # We make the sum on all runs
        test_failure_rate[i] += result[0][i]
        training_failure_rate[i] += result[1][i]
        print(test_failure_rate)
        print(training_failure_rate)

for i in range(len(test_failure_rate)) :
    # We want an average but only made the sum by now
    test_failure_rate[i] /= runs
    training_failure_rate[i] /= runs


print(test_failure_rate)
print("-----")
print(training_failure_rate)

mnist.results([i / 60000 for i in iterations], training_failure_rate, test_failure_rate, runs)
