#! /usr/bin/python3
# -*- coding:utf-8 -*-

from multiprocessing import Pool
import mnist
import numpy as np

# Number of processors
processors = 4

# Max times
Tmax = 8


training_image = mnist.read_idx("data/MNIST/train-images-idx3-ubyte")
training_label = mnist.read_idx("data/MNIST/train-labels-idx1-ubyte")
test_image = mnist.read_idx("data/MNIST/t10k-images-idx3-ubyte")
test_label = mnist.read_idx("data/MNIST/t10k-labels-idx1-ubyte")

test_failure_rate = [0] * Tmax
training_failure_rate = [0] * Tmax

def doUrStuff(T) :
    global training_image
    global training_label
    global test_image
    global test_label
    global test_failure_rate
    global training_failure_rate
    for t in T :
        a,b = mnist.main(t, training_image, training_label, test_image, test_label)
        test_failure_rate[t-1] = a
        training_failure_rate[t-1] = b
    return test_failure_rate, training_failure_rate

pool = Pool(processes = processors)

# distributing times_tab to every core
times = [i for i in range(1, Tmax + 1)]
list_times_tab = tuple([[n, Tmax+1-n] for n in range(1,Tmax // 2 + 1)]) # all elements of this list are equivalent in terms of time complexity

results = pool.map(doUrStuff, list_times_tab)

for result in results :
    for i in range(len(result[0])) :
        test_failure_rate[i] += result[0][i]
        training_failure_rate[i] += result[1][i]

print(test_failure_rate)
print("-----")
print(training_failure_rate)

mnist.results(times, training_failure_rate, test_failure_rate)
locate_min = test_failure_rate.index(min(test_failure_rate))
print("best:", locate_min)
