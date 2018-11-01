from network import Network
from neurone import Layer

import optimiser as opt
import activation as act
from outils import read_idx

from random import shuffle
from concurrent.futures import ProcessPoolExecutor, Future
import csv

training_images = read_idx("data/MNIST/train-images-idx3-ubyte")
training_labels = read_idx("data/MNIST/train-labels-idx1-ubyte")
test_images = read_idx("data/MNIST/t10k-images-idx3-ubyte")
test_labels = read_idx("data/MNIST/t10k-labels-idx1-ubyte")

N_CPUS = 4


def save_to_csv(filename, legend, results_list):
    with open(filename, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(legend)
        csv_writer.writerows(results_list)


def create_SGD_network(learning_rate):
    network = Network([
        Layer(784, 16, act.regulated_tanh),
        Layer(16, 16, act.regulated_tanh),
        Layer(16, 10, act.regulated_tanh)
    ], normalisation=True, optimiser=opt.SGD(learning_rate))
    return network


def create_RMS_network(learning_rate, decay, epsilon=10**(-8)):
    network = Network([
        Layer(784, 16, act.regulated_tanh),
        Layer(16, 16, act.regulated_tanh),
        Layer(16, 10, act.regulated_tanh)
    ], normalisation=True, optimiser=opt.RMSprop(learning_rate, decay, epsilon=epsilon))
    return network


def train(network, training_images, training_labels, indices):
    for i in indices:
        image, label = training_images[i], training_labels[i]
        lb = [1 if int(label) == p else 0 for p in range(10)]
        network.learning(image.flatten(), lb)


def error_rate(network, test_images, test_labels, indices=None):
    if indices is None:
        indices = range(len(test_images))
    success, errors = 0, 0
    for i in indices:
        image, label = test_images[i], test_labels[i]
        output = network.test(image.flatten())
        most_likely = output.argmax()
        if int(label) == most_likely:
            success += 1
        else:
            errors += 1
    assert success + errors == len(indices)
    return errors / len(indices)


def run(name, nb_batch, network):
    global training_images
    global training_labels
    global test_images
    global test_labels
    points_per_batch = 2
    batch_size = len(training_images)
    nb_train = batch_size // points_per_batch
    i = 0
    rate = error_rate(network, test_images, test_labels)
    print(f"{name} : iteration 0 rate {rate}")
    result = ([0], [rate])
    indices = list(range(batch_size))
    for j in range(nb_batch):
        shuffle(indices)
        for i in range(points_per_batch):
            train(network, training_images, training_labels,
                  indices[i*nb_train: (i+1)*nb_train])
            rate = error_rate(network, test_images, test_labels)
            result[0].append(j*batch_size + (i+1)*nb_train)
            result[1].append(rate)
            print(f"{name} : iteration {j*batch_size + (i+1)*nb_train} rate {rate}")
    return result


# network = create_RMS_network(0.003)
# results = run(1, 3, network)
# print(results[1])

def multi_run():
    n_runs = 2
    nb_batchs = 2
    with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
        results_sgd = []
        for i in range(n_runs):
            network = create_SGD_network(0.003)
            results_sgd.append(executor.submit(
                run, f"sgd {i}", nb_batchs, network))
        results_rms = []
        for i in range(n_runs):
            network = create_RMS_network(0.003, 0.9, 10**(-8))
            results_rms.append(executor.submit(
                run, f"rms {i}", nb_batchs, network))
        results_sgd = list(map(Future.result, results_sgd))
        results_rms = [result.result() for result in results_rms]
    save_to_csv('data/RMS2prop.csv',
                results_rms[0][0], (res[1] for res in results_rms))
    save_to_csv('data/SGD2.csv',
                results_sgd[0][0], (res[1] for res in results_sgd))


if __name__ == "__main__":
    multi_run()
