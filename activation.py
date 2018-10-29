import numpy as np
from abc import ABC, abstractclassmethod


class Activation(ABC):

    def __init__(self): ...

    @abstractclassmethod
    def f(cls, x): ...

    @abstractclassmethod
    def df(cls, x): ...


class sigmoid(Activation):
    @classmethod
    def f(cls, x):
        return 1 / (1 + np.exp(-x))

    @classmethod
    def df(cls, x):
        return cls.f(x) * (1 - cls.f(x))


class tanh(Activation):
    @classmethod
    def f(cls, x):
        return np.tanh(x)

    @classmethod
    def df(cls, x):
        return 1 - (cls.f(x) ** 2)


class regulated_tanh(Activation):
    @classmethod
    def f(cls, x):
        return 1.7159 * np.tanh(x * (2.0 / 3.0))

    @classmethod
    def df(cls, x):
        return 1.7159 * (2.0 / 3.0) * (1 - (np.tanh(x * (2.0 / 3.0)) ** 2))


class softmax(Activation):
    @classmethod
    def f(cls, x):
        # needed to not overflow on great inputs, does not change the result
        shiftx = x - np.max(x)

        exp = np.exp(shiftx)
        return exp / exp.sum()

    @classmethod
    def df(cls, x):
        tmp = cls.f(x)
        return tmp - tmp**2
