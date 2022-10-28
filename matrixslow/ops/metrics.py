import abc

import numpy as np

from ..core import Node


class Metrics(Node):
    def __init__(self, *inputs, name='metrics'):
        super().__init__(name)
        self.inputs = list(inputs)
        self.set_output()
        self.init()

    @abc.abstractmethod
    def init(self):
        pass

    def reset(self):
        self.reset_values()
        self.init()

    @staticmethod
    def prob_to_label(prob, thresholds=0.5):
        if prob.shape[0] > 1:
            labels = np.zeros((prob.shape[0], 1))
            labels[np.argmax(prob, axis=0)] = 1
        else:
            labels = np.where(prob < thresholds, -1, 1)
        return labels

    def get_jacobi(self, input_node):
        raise NotImplementedError()

    def value_str(self):
        return f"{self.__class__.__name__}: {self.value:.4f}"


class Accuracy(Metrics):
    def __init__(self, *inputs, name='accuracy'):
        super().__init__(*inputs, name=name)
        self.correct_num = 0
        self.total_num = 0

    def init(self):
        self.correct_num = 0
        self.total_num = 0

    def compute(self):
        pred = Metrics.prob_to_label(self.inputs[0].value)
        labels = self.inputs[1].value
        if pred.shape[0] > 1:
            self.correct_num += np.sum(np.multiply(pred, labels))
            self.total_num += pred.shape[1]
        else:
            self.correct_num += np.sum(pred == labels)
            self.total_num += len(pred)
        self.value = self.correct_num / self.total_num


