#! /usr/bin/python3
# -*- coding:utf-8 -*-

import dill as pickle
import matplotlib.pyplot as plt
import math

def plot(filename, plot_standard_deviation, period_of_std) :
    # plot_standard_deviation is a boolean : whether or not to show the standard deviation
    # period of std refers to the number of points between each standard deviation appears
    X = pickle.load(open(filename, "rb"))
    iterations = X["abscisse"]
    training_failure_rate = X["training rate"]
    test_failure_rate = X["test rate"]
    runs = X["runs"]
    training_failure_deviation = X["training deviation"]
    test_failure_deviation = X["test deviation"]

    plt.xlabel("Number of iterations")
    plt.ylabel("Average failure rate (on {} runs)".format(runs))
    plt.plot(iterations, training_failure_rate,'r-', label="Failure rate on the training sample")
    plt.plot(iterations, test_failure_rate, 'b-', label="Failure rate on the test sample")
    plt.legend()
    if plot_standard_deviation :
        plt.title("Failure rates of the network on training and test samples ({} runs, Condidence level : 95%)".format(runs))
        plt.errorbar([iterations[i] for i in range(len(iterations)) if i % period_of_std == 0], [training_failure_rate[i] for i in range(len(training_failure_deviation)) if i % period_of_std == 0], [2 * training_failure_deviation[i] / math.sqrt(runs) for i in range(len(training_failure_deviation)) if i % period_of_std == 0], color="red",linestyle="None")
        plt.errorbar([iterations[i] for i in range(len(iterations)) if i % period_of_std == 0], [test_failure_rate[i] for i in range(len(test_failure_deviation)) if i % period_of_std == 0], [2 * test_failure_deviation[i] / math.sqrt(runs) for i in range(len(test_failure_deviation)) if i % period_of_std == 0], color="blue",linestyle="None")
    else :
        plt.title("Failure rates of the network on training and test samples ({} runs)".format(runs))
    plt.savefig("MNIST.png", transparent=False)
    plt.savefig("MNIST_trans.png", transparent=True)
    plt.show()
