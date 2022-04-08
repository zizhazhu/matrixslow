import abc

import numpy as np

class Node:

    def __init__(self):
        self.value = None
        self.jacobi = None
        self.inputs = []
        self.outputs = set()
        self._shape = None

    def set_output(self):
        for input_unit in self.inputs:
            input_unit.outputs.add(self)

    def __repr__(self):
        return repr(self.value)

    def forward(self):
        for node in self.inputs:
            if node.value is None:
                node.forward()
        self.compute()

    @abc.abstractmethod
    def compute(self):
        raise NotImplementedError()

    def backward(self, result):
        """
        :param result: target
        :return: gradient
        """
        if self.jacobi is None:
            if self is result:
                # start from 1
                self.jacobi = np.eye(self.shape[0], self.shape[1])
            else:
                self.jacobi = np.zeros(self.shape)
                for output in self.outputs:
                    if output.value is not None:
                        gradient = output.backward(result)
                        jacobi = output.get_jacobi(self)
                        self.jacobi += gradient * jacobi
        return self.jacobi

    @abc.abstractmethod
    def get_jacobi(self, input_node):
        raise NotImplementedError()

    @property
    def shape(self):
        return self._shape


