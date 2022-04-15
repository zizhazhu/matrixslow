import numpy as np

from ..core import Node


class Operator(Node):
    pass


class Add(Operator):
    def __init__(self, a, b):
        super().__init__()
        self.inputs = [a, b]
        self.value = None
        self.set_output()

    def compute(self):
        shape = self.inputs[0].shape
        self.value = np.mat(np.zeros(shape))
        for input_unit in self.inputs:
            self.value += input_unit.value

    def get_jacobi(self, input_node):
        return np.eye(self.dimension)
        

class MatMul(Operator):
    def __init__(self, a, b):
        super().__init__()
        self.inputs = [a, b]
        self.value = None
        self.set_output()

    def compute(self):
        self.value = np.matmul(self.inputs[0].value, self.inputs[1].value)

    def get_jacobi(self, input_node):
        # matrix multiplication jacobi
        zeros = np.zeros((self.dimension, input_node.dimension))
        if input_node is self.inputs[0]:
            return fill_diagonal(zeros, self.inputs[1].value.T)
        else:
            jacobi = fill_diagonal(zeros, self.inputs[0].value)
            row_sort = np.arange(self.dimension).reshape(self.shape[::-1]).T.ravel()
            col_sort = np.arange(input_node.dimension).reshape(input_node.shape[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


def fill_diagonal(to_be_filled, filler):
    factor = int(to_be_filled.shape[0] / filler.shape[0])
    m, n = filler.shape
    for i in range(factor):
        to_be_filled[i*m:(i+1)*m, i*n:(i+1)*n] = filler
    return to_be_filled