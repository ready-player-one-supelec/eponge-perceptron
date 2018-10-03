import lib
import numpy as np 
import matplotlib.pyplot as plt

samples = [
    (np.array([[0],[0]]), np.array([[0]])),
    (np.array([[1],[0]]), np.array([[1]])),
    (np.array([[0],[1]]), np.array([[1]])),
    (np.array([[1],[1]]), np.array([[0]]))
]

model = lib.model.Model([
    lib.layer.InputLayer(2, bias=True),
    lib.layer.Layer(2, activation=lib.activation.tanh, bias=True),
    lib.layer.Layer(1, activation=lib.activation.tanh),
])

model.initialize_random()

for elem in samples:
    infered = model.infer(elem[0])
    print(infered, model.loss(infered, elem[1]))

print('------------')
for i in range(10000):
    for j, elem in enumerate(samples):
        infered = model.infer(elem[0])
        model.update_weights(infered, elem[1])

for elem in samples:
    infered = model.infer(elem[0])
    print(infered, model.loss(infered, elem[1]))

N = 100
# print(model.infer(np.array([[0.5],[0.5]], dtype=np.float_))[0,0])
values = [[model.infer(np.array([[i],[j]], dtype=np.float_)/N)[0, 0] for i in range(N + 1)] for j in range(N+1)]
# print(values)
plt.imshow(values, extent=[0,1,0,1], cmap='gray')
plt.show()