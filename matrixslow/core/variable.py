import numpy as np

from .node import Node


class Variable(Node):

    def __init__(self, dim, init=None, trainable=True, **kwargs):
        super().__init__()
        self.dim = dim
        # TODO: use initializer
        if init:
            self.value = np.mat(np.random.normal(0, 0.001, self.dim))
        self.trainable = trainable

    def set_value(self, value):
        self.reset_values()
        self.value = value

    def compute(self):
        pass
