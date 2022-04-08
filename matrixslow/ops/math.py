import numpy as np

from ..core import Node


class Operator(Node):
    pass


class Add(Operator):
    def __init__(self, a, b):
        super().__init__()
        self.inputs = [a, b]
        self._shape = self.inputs[0].shape
        self.value = None
        self.set_output()

    def compute(self):
        self.value = np.mat(np.zeros(self.shape))
        for input_unit in self.inputs:
            self.value += input_unit.value

    def get_jacobi(self, input_node):
        return np.eye(self.shape)
        

class MatMul(Operator):
    def __init__(self, a, b):
        super().__init__()
        self.inputs = [a, b]
        self.value = None
        self.set_output()

    def compute(self):
        self.value = np.matmul(self.inputs[0].value, self.inputs[1].value)

    def get_jacobi(self, input_node):
        raise NotImplementedError()

    @property
    def shape(self):
        return self.inputs[0].shape[0], self.inputs[1].shape[1]