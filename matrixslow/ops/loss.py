import numpy as np

from ..core import Node


class LossFunction(Node):
    pass


class PerceptionLoss(LossFunction):
    def __init__(self, input_value):
        super().__init__()
        self.inputs.append(input_value)
        self.set_output()

    def compute(self):
        self.value = np.mat(np.where(
            self.inputs[0].value >= 0.0, 0.0, -self.inputs[0].value
        ))

    def get_jacobi(self, input_node):
        diag = np.where(input_node.value >= 0.0, 0.0, -1.0)
        return np.diag(diag.ravel())
