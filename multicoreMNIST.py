#! /usr/bin/python3
# -*- coding:utf-8 -*-

from multiprocessing import Pool
import mnist
import numpy as np
import dill as pickle
import os
from outils import *
import random

# IMPORTANT
# Forcing numpy to be monothread to avoid over-threading
# DO -> export OPENBLAS_NUM_THREADS=1 before launching the script

# Number of processors
processors = 4

# Number of runs (for the average)
runs = 4

# Times we learn on the training sample
step = 30000 # we want to make a point even if the whole training set has not been coped with yet
number_of_complete_iterations = 5
iterations = create_range(0, number_of_complete_iterations * 60000 - 1, step)
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


test_results = [result[0] for result in results]
training_results = [result[1] for result in results]

# We want the average and the standard deviation
test_failure_rate = np.mean(test_results, axis=0)
training_failure_rate = np.mean(training_results, axis=0)
test_failure_deviation = np.std(test_results, axis=0)
training_failure_deviation = np.std(training_results, axis=0)


X = {"abscisse" : [i/60000 for i in iterations], "training rate" : training_failure_rate, "test rate": test_failure_rate, "runs" : runs, "training deviation" : training_failure_deviation, "test deviation": test_failure_deviation}
pickle.dump(X, open("tmp", "wb"))
