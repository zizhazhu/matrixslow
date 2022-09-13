import numpy as np

from ..core import Node


class Operator(Node):
    def __init__(self):
        super().__init__()


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
            # 调整矩阵顺序，让第一个矩阵的列对齐梯度矩阵的列（这两者维度相同）
            jacobi = fill_diagonal(zeros, self.inputs[0].value)
            row_sort = np.arange(self.dimension).reshape(self.shape[::-1]).T.ravel()
            col_sort = np.arange(input_node.dimension).reshape(input_node.shape[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class Multiply(Operator):
    def __init__(self, a, b):
        super().__init__()
        self.inputs = [a, b]
        self.set_output()

    def compute(self):
        self.value = np.multiply(self.inputs[0].value, self.inputs[1].value)

    def get_jacobi(self, input_node):
        if input_node is self.inputs[0]:
            return np.diag(self.inputs[1].value.A1)
        else:
            return np.diag(self.inputs[0].value.A1)


class Step(Operator):

    def __init__(self, a):
        super().__init__()
        self.inputs = [a]
        self.value = None
        self.set_output()

    def compute(self):
        self.value = np.where(self.inputs[0].value >= 0, 1.0, 0.0)

    def get_jacobi(self, input_node):
        return np.zeros((self.dimension, input_node.dimension))


class Sigmoid(Operator):

    def __init__(self, a):
        super().__init__()
        self.inputs = [a]
        self.set_output()

    def compute(self):
        x = self.inputs[0].value
        self.value = np.mat(1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))))

    def get_jacobi(self, parent):
        # 展开成1维Array再对角化
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)


class Softmax(Operator):

    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def __init__(self, a):
        super().__init__()
        self.inputs = [a]
        self.set_output()

    def compute(self):
        self.value = Softmax.softmax(self.inputs[0].value)

    def get_jacobi(self, input_node):
        raise NotImplementedError("Don't use SoftMax's get_jacobi")


def fill_diagonal(to_be_filled, filler):
    factor = int(to_be_filled.shape[0] / filler.shape[0])
    m, n = filler.shape
    for i in range(factor):
        to_be_filled[i*m:(i+1)*m, i*n:(i+1)*n] = filler
    return to_be_filled