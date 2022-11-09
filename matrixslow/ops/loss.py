import numpy as np

from ..core import Node
from .math import *


class LossFunction(Node):
    pass


class PerceptionLoss(LossFunction):
    def __init__(self, input_value, name='perception_loss'):
        super().__init__(name=name)
        self.inputs.append(input_value)
        self.set_output()

    def compute(self):
        # 当分类正确时，值大于0，分类错误时小于0，所以只在分类错误时有损失
        self.value = np.mat(np.where(
            self.inputs[0].value >= 0.0, 0.0, -self.inputs[0].value
        ))

    def get_jacobi(self, input_node):
        # 参考值的计算，只在分类错误时有损失，也才有梯度，梯度是-x求导
        diag = np.where(input_node.value >= 0.0, 0.0, -1.0)
        return np.diag(diag.ravel())


class LogLoss(LossFunction):

    def __init__(self, input_value, name='logloss'):
        super().__init__(name)
        self.inputs = [input_value]
        self.set_output()

    def compute(self):
        x = self.inputs[0].value
        self.value = np.mat(np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))))

    def get_jacobi(self, input_node):
        x = input_node.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))
        return np.diag(diag.ravel())


class CrossEntropyWithSoftMax(LossFunction):

    def __init__(self, logits, label, name='crossentropy_loss'):
        super().__init__(name=name)
        self.inputs = [logits, label]
        self.set_output()

    def compute(self):
        prob = Softmax.softmax(self.inputs[0].value)
        self.value = np.mat(
            -np.sum(np.multiply(self.inputs[1].value, np.log(prob + 1e-12)))
        )

    def get_jacobi(self, input_node):
        prob = Softmax.softmax(self.inputs[0].value)
        if input_node is self.inputs[0]:
            return (prob - self.inputs[1].value).T
        else:
            return (-np.log(prob + 1e-12)).T
