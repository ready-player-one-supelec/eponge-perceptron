#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def get_data():
    
    # number of layers (must be superior or equal to 2)
    l = 3
    
    # number of neurons for each layer
    S = [2, 4, 1]
    
    # activation function and its derivative for each layer
    sig = lambda x:1/(1+np.exp(-x))
    sigp = lambda x:sig(x)*(1-sig(x)) # attention sig(x) déjà calculé > TODO
    tanh = lambda x:1.7159*np.tanh(2/3*x)
    tanhp = lambda x:1-np.tanh(x)**2
    F = [sig, sig, sig] # activation functions
    Fp = [sigp, sigp, sigp] # derived activation functions
    lr = 0.5 # learning rate
    
    # XOR training  (assumed all data have the same size)
    training_data = [
        (np.array([[0],[0]]), np.array([[0]])),
        (np.array([[0],[1]]), np.array([[1]])),
        (np.array([[1],[0]]), np.array([[1]])),
        (np.array([[1],[1]]), np.array([[0]]))
        ]
    
    nb_pass = 5000 # number of pass throught the entire training set
    
    # tests :
    # the number of layer must be superior or equal to 2,
    # we must get the number of neurons, the activation function and its derivative for all the layers,
    # the number of neurons of the first layer must be equal to the number of inputs,
    # the number of neurons of the last layer must be equal to the number of outputs.
    
    if l>=2 and len(S)==l and len(F)==l and len(Fp)==l and len(training_data[0][0])==S[0] and len(training_data[0][1])==S[-1]:
        return l, S, F, Fp, lr, training_data, nb_pass
    else:
        print('---\ninput data error\n---')
        return None
    
    
def init_weights_and_biases(l, S):
    
    # initialise weights
    Splus = [S[0]] + S # inputs are considered like a layer
    W = [np.zeros((Splus[layer+1], Splus[layer])) for layer in range(l)]
    
    # weights are randomly drawn from an uniform distribution with mean zero and standart deviation m**0.5 where m is the fan-in (the number of connections feeding into the node)
    for layer in range(l):
        for neuron in range(S[layer]):
            m = Splus[layer] # fan-in
            W[layer][neuron] = np.random.uniform(-(3/m)**0.5, (3/m)**0.5, (1,Splus[layer]))
    
    # initialise biases
    B = [np.zeros((S[k], 1)) for k in range(l)]
    # same method as the weights
    for layer in range(l):
        for neuron in range(S[layer]):
            m = Splus[layer] # fan-in
            B[layer][neuron] = np.random.uniform(-(3/m)**0.5, (3/m)**0.5, (1,1))

    return W, B


def init_temporary_arrays(l, S):
    
    # sum of the inputs for each neuron
    N = [np.zeros((S[layer], 1)) for layer in range(l)]
    
    # output of the activation function for each neuron
    Splus = [S[0]] + S # inputs are considered like a layer
    A = [np.zeros((Splus[layer], 1)) for layer in range(l+1)]
    
    return N, A


def prop(l, S, F, W, B, N, A, P):
    
    A[0] = P # inputs are considered like a layer
    
    for layer in range(l):
        N[layer] = W[layer] @ A[layer] - B[layer]
        A[layer+1] = F[layer](N[layer])
        
    return N, A
    

def backprop(l, Fp, lr, W, B, N, A, D):
    
    error = D-A[l]
    total_error = 0.5 * np.sum(np.transpose(error) @ error) #*2
    
    Sa = -np.diag(Fp[l-1](np.transpose(N[l-1]))[0]) @ error #*2
    
    for layer in range(l-1, 0, -1):
        
        Sb = np.diag(Fp[layer-1](np.transpose(N[layer-1]))[0]) @ np.transpose(W[layer]) @ Sa #*2
        
        # update weights
        W[layer] = W[layer] - lr * Sa @ np.transpose(A[layer])
        # update biases
        B[layer] = B[layer] + lr * Sa
        
        Sa = Sb
        
    # update weights
    W[0] = W[0] - lr * Sa @ np.transpose(A[0])
    # update biases
    B[0] = B[0] + lr * Sa
        
    return W, B, total_error


def single_training_set(l, S, F, Fp, lr, W, B, N, A, P, D):
    
    N, A = prop(l, S, F, W, B, N, A, P)
    W, B, total_error = backprop(l, Fp, lr, W, B, N, A, D)
    
    return W, B, total_error


def stochastic_method(l, S, F, Fp, lr, training_data, nb_pass, W=None, B=None):
    
    if W==None and B==None:
        W, B = init_weights_and_biases(l, S)
    elif W==None or B==None:
        print('---\ninitial data error\n---')
        return None
    # else: W, A are defined
    
    N, A = init_temporary_arrays(l, S)
    
    total_errors = [0]*nb_pass*len(training_data)
    
    num = 0
    while num < nb_pass*len(training_data):
        
        # shuffling training set
        np.random.shuffle(training_data)
        
        for (P, D) in training_data:
            
            W, B, total_error = single_training_set(l, S, F, Fp, lr, W, B, N, A, P, D)
            
            total_errors[num] = total_error
            num += 1
            
    return W, B, total_errors


def plot_result(total_errors):
    
    plt.plot(total_errors)
    plt.xlabel('Number of learning rounds')
    plt.ylabel('Error')
    
    
def launch(W=None, B=None, get_values=0):
    
    l, S, F, Fp, lr, training_data, nb_pass = get_data()
    W, B, total_errors = stochastic_method(l, S, F, Fp, lr, training_data, nb_pass, W, B)
    
    plot_result(total_errors)
    
    if get_values:
        return W, B, total_errors
    
    
def test(W, B):
    
    l, S, F, Fp, lr, training_data, nb_pass = get_data()
    N, A = init_temporary_arrays(l, S)
    
    for (P, D) in training_data:
        N, A = prop(l, S, F, W, B, N, A, P)
        print(A[l],D)