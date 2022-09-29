import numpy as np

from .node import Node


class Variable(Node):

    def __init__(self, dim, init=None, trainable=True, name='var'):
        super().__init__(name)
        self.dim = tuple(dim)
        # TODO: use initializer
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))
        self.trainable = trainable

    def set_value(self, value):
        self.reset_values()
        if value.shape != self.dim:
            raise ValueError(f'Variable need shape {self.dim} not {value.shape}')
        self.value = value

    def compute(self):
        pass

    @property
    def shape(self):
        return self.dim
