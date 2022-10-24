import numpy as np

from matrixslow.core import Node


class Operator(Node):
    def __init__(self, name='op'):
        super().__init__(name)


class Reshape(Operator):

    def __init__(self, node, shape):
        super().__init__()
        self.inputs = [node]
        self.to_shape = shape
        self.set_output()

    def compute(self):
        value = self.inputs[0].value
        self.value = value.reshape(self.to_shape)

    def get_jacobi(self, input_node):
        return np.mat(np.eye(self.dimension))


class Concat(Operator):

    def __init__(self, nodes, axis=0):
        super().__init__()
        self.inputs = list(nodes)
        self.axis = axis
        self.set_output()

    def compute(self):
        self.value = np.concatenate(
            [node.value.flatten() for node in self.inputs], axis=self.axis,
        ).T

    def get_jacobi(self, input_node):
        dimensions = [p.dimension for p in self.inputs]
        pos = self.inputs.index(input_node)
        dimension = input_node.dimension

        jacobi = np.mat(np.zeros((self.dimension, dimension)))
        start_row = int(np.sum(dimensions[:pos]))
        jacobi[start_row:start_row+dimension,0:dimension] = np.eye(dimension)

        return jacobi

