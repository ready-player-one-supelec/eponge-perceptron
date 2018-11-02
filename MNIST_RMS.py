from network import Network
from neurone import Layer

import optimiser as opt
import activation as act
from outils import read_idx

from random import shuffle
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime

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


def create_SGD_builder(learning_rate):
    def builder():
        network = Network([
            Layer(784, 16, act.regulated_tanh),
            Layer(16, 16, act.regulated_tanh),
            Layer(16, 10, act.regulated_tanh)
        ], normalisation=True, optimiser=opt.SGD(learning_rate))
        return network
    return builder


def create_RMS_builder(learning_rate, decay, epsilon=10**(-8)):
    def builder():
        network = Network([
            Layer(784, 16, act.regulated_tanh),
            Layer(16, 16, act.regulated_tanh),
            Layer(16, 10, act.regulated_tanh)
        ], normalisation=True,
            optimiser=opt.RMSprop(learning_rate, decay, epsilon=epsilon)
        )
        return network
    return builder


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

def multi_run(network_builders, n_runs, nb_batchs):
    now = datetime.now()
    with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
        results_list = [[] for _ in network_builders]
        names = ['' for _ in network_builders]
        for p, (network_builder, results) in enumerate(
            zip(network_builders, results_list)
        ):
            for i in range(n_runs):
                network = network_builder()
                results.append(executor.submit(
                    run, f"{network} {i}", nb_batchs, network))
            names[p] = str(network)
        for results, name in zip(results_list, names):
            print(name)
            results = [result.result() for result in results]
            save_to_csv(f'data/{name}-{str(now).replace(" ", "")}.csv',
                        results[0][0], (res[1] for res in results))


if __name__ == "__main__":
    multi_run([
        create_SGD_builder(0.003),
        create_RMS_builder(0.003, 0.9, 10**(-8))
    ],
        n_runs=1, nb_batchs=0
    )
