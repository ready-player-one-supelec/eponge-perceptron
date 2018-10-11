import lib
import numpy as np 
import matplotlib.pyplot as plt
import csv

#variables definition
samples = [
    (np.array([[0],[0]]), np.array([[0]])),
    (np.array([[1],[0]]), np.array([[1]])),
    (np.array([[0],[1]]), np.array([[1]])),
    (np.array([[1],[1]]), np.array([[0]]))
]
n_batches = 5000
n_trainings = 100

def train_whith_hidden_layer_size(size):
    losses_list = [[] for i in range(n_trainings)]
    result_file = f'xor_tanh_training_nerones{size}.csv'
    # model definition
    model = lib.model.Model([
        lib.layer.InputLayer(2, bias=True),
        lib.layer.Layer(size, activation=lib.activation.tanh, bias=True),
        lib.layer.Layer(1, activation=lib.activation.tanh),
    ])
    model.initialize_random()
    #training
    print('------------')
    for k, losses in enumerate(losses_list):
        for i in range(n_batches):
            batch_loss = 0.0
            for j, elem in enumerate(samples):
                infered = model.infer(elem[0])
                model.update_weights(infered, elem[1])
                batch_loss += model.loss(infered, elem[1])
            losses.append(batch_loss / len(samples))
        model.initialize_random()
        print(f"doing training {k} of {n_trainings}")

    #storing results
    with open(result_file, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(losses_list)

for size in [2,3,5,7]:
    train_whith_hidden_layer_size(size)