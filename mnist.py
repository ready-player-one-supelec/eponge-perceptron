import lib
import numpy as np
from random import shuffle

test = (lib.utils.read_idx('datasets/t10k-images.idx3-ubyte'), lib.utils.read_idx('datasets/t10k-labels.idx1-ubyte'))
train = (lib.utils.read_idx('datasets/train-images.idx3-ubyte'), lib.utils.read_idx('datasets/train-labels.idx1-ubyte'))

def labels_to_vec(label):
    return (np.arange(0,10) == label).astype(np.float_)

def list_shuffle(xs):
    shuffle(xs)
    return xs

def show(image):
    for i in range(image.shape[1]+2):
        print('&', end='')
    print()
    for row in image:
        print('&', end='')
        for value in row:
            print('+' if value>0 else ' ', sep='', end='')
        print('&')
    for i in range(image.shape[1]+1):
        print('&', end='')
    print()

#image flattening
# test[0].shape = (test[0].shape[0], -1,1)
# train[0].shape = (train[0].shape[0], -1,1)


model = lib.model.Model([
    lib.layer.InputLayer(28*28, bias=True),
    lib.layer.Layer(1000,lib.activation.tanh, bias=True),
    lib.layer.Layer(10,lib.activation.tanh)
])
print()

model.initialize_random()

mean_loss = 0
for i in list_shuffle(list(range(test[0].shape[0]))):
    output = model.infer(test[0][i].reshape(-1,1))
    model.update_weights(output, labels_to_vec(test[1][i]).reshape(-1,1))
    if i % 1000 == 0:
        loss = model.loss(output,labels_to_vec(test[1][i]).reshape(-1,1))
        print(loss)
        # show(test[0][i])