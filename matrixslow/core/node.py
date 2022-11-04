import abc

import numpy as np

from .graph import default_graph


class Node:

    def __init__(self, name=None, need_save=True):
        self.value = None
        self.grad = None
        self.inputs = []
        self.outputs = set()
        self._shape = None

        self.need_save = need_save

        self.graph = default_graph
        if name is None:
            name = self.__class__.__name__
        new_name = self.graph.add_node(self, name)
        self._name = new_name

    def set_output(self):
        # 把当前节点标记到输入节点的输出节点
        for input_unit in self.inputs:
            input_unit.outputs.add(self)

    def __repr__(self):
        return f'{self._name}: {repr(self.value)}'

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
        if self.grad is None:
            if self is result:
                # start from [[1]]
                self.grad = np.mat(np.eye(self.dimension))
            else:
                self.grad = np.mat(np.zeros((result.dimension, self.dimension)))
                for output in self.outputs:
                    if output.value is not None:
                        # 后继节点积累下来的梯度
                        gradient = output.backward(result)
                        # 后继节点到当前节点的梯度
                        jacobi = output.get_jacobi(self)
                        self.grad += gradient * jacobi
        return self.grad

    @abc.abstractmethod
    def get_jacobi(self, input_node):
        raise NotImplementedError()

    @property
    def shape(self):
        return self.value.shape

    @property
    def dimension(self):
        return self.value.shape[0] * self.value.shape[1]

    def clear_jacobi(self):
        self.grad = None

    def reset_values(self, recursive=True):
        self.value = None
        if recursive:
            for output_unit in self.outputs:
                output_unit.reset_values(recursive)

    @property
    def kwargs(self):
        return {'name': self._name}

