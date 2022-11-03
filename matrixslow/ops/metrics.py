import abc

import numpy as np

from ..core import Node


class Metrics(Node):
    def __init__(self, *inputs, name='metrics'):
        super().__init__(name)
        self.inputs = list(inputs)
        self.set_output()

    @abc.abstractmethod
    def init(self):
        pass

    def reset(self):
        self.reset_values()
        self.init()

    @staticmethod
    def prob_to_label(prob, thresholds=0.5):
        # probability to 0,1 label
        if prob.shape[0] > 1:
            labels = np.zeros((prob.shape[0], 1))
            labels[np.argmax(prob, axis=0)] = 1
        else:
            labels = np.where(prob < thresholds, -1, 1)
        return labels

    def get_jacobi(self, input_node):
        raise NotImplementedError()

    def value_str(self):
        return f"{self.__class__.__name__}: {self.value:.4f} "


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


class Precision(Metrics):
    def __init__(self, *inputs, name='precision'):
        super().__init__(*inputs, name=name)
        self.true_pos_num = 0
        self.pred_pos_num = 0

    def init(self):
        self.true_pos_num = 0
        self.pred_pos_num = 0

    def compute(self):
        pred = Metrics.prob_to_label(self.inputs[0].value)
        labels = np.array(self.inputs[1].value)
        self.pred_pos_num += np.sum(pred == 1, axis=1)
        self.true_pos_num += np.sum(np.logical_and(pred == labels, labels == 1), axis=1)
        self.value = np.divide(self.true_pos_num, self.pred_pos_num)

    def value_str(self):
        return f"{self.__class__.__name__}: {self.value} "


class Recall(Metrics):
    def __init__(self, *inputs, name='recall'):
        super().__init__(*inputs, name=name)
        self.true_pos_num = 0
        self.true_num = 0

    def init(self):
        self.true_pos_num = 0
        self.true_num = 0

    def compute(self):
        pred = Metrics.prob_to_label(self.inputs[0].value)
        labels = self.inputs[1].value
        self.true_num += np.sum(labels == 1)
        self.true_pos_num += np.sum(np.multiply(pred, labels) == 1)
        if self.true_num == 0:
            self.value = 0
        else:
            self.value = self.true_pos_num / self.true_num


class ROC(Metrics):
    def __init__(self, *inputs, count=100, name='roc'):
        super().__init__(*inputs, name=name)
        self.count = count
        self.positive_count = 0
        self.negative_count = 0
        self.fpr = np.array([0.0] * count)
        self.tpr = np.array([0.0] * count)
        self.true_pos_count = np.array([0] * count)
        self.false_pos_count = np.array([0] * count)

    def init(self):
        self.positive_count = 0
        self.negative_count = 0
        self.fpr = np.array([0.0] * self.count)
        self.tpr = np.array([0.0] * self.count)
        self.true_pos_count = np.array([0] * self.count)
        self.false_pos_count = np.array([0] * self.count)

    def compute(self):
        prob = self.inputs[0].value
        labels = self.inputs[1].value
        self.positive_count += np.sum(labels == 1)
        self.negative_count += np.sum(labels == -1)

        thresholds = np.linspace(0, 1, self.count)

        for i in range(thresholds.size):
            pred = Metrics.prob_to_label(prob, thresholds[i])
            self.true_pos_count[i] += np.sum(pred == 1 and labels == 1)
            self.false_pos_count[i] += np.sum(pred == 1 and labels == -1)

        if self.positive_count > 0 and self.negative_count > 0:
            self.tpr = self.true_pos_count / self.positive_count
            self.fpr = self.false_pos_count / self.negative_count


class AUC(Metrics):
    def __init__(self, *inputs, name='auc'):
        self.roc = ROC(*inputs, count=100)
        super().__init__(*inputs, name=name)

    def init(self):
        self.roc.init()

    def compute(self):
        self.roc.compute()
        self.value = np.sum((self.roc.tpr[:-1] - self.roc.tpr[1:]) * (1 - self.roc.fpr[1:]))

    def value_str(self):
        return f"{self.__class__.__name__}: {self.value:.4f}"


class F1Score(Metrics):
    def __init__(self, *inputs, name='f1_score'):
        super().__init__(*inputs, name=name)
        self.true_pos_count = 0
        self.positive_count = 0
        self.pred_pos_count = 0

    def init(self):
        self.true_pos_count = 0
        self.positive_count = 0
        self.pred_pos_count = 0

    def compute(self):
        pred = Metrics.prob_to_label(self.inputs[0].value)
        labels = self.inputs[1].value

        self.positive_count += np.sum(labels == 1)
        self.pred_pos_count += np.sum(pred == 1)
        self.true_pos_count += np.sum(pred == 1 and labels == 1)

        if self.pred_pos_count > 0:
            precision = self.true_pos_count / self.pred_pos_count
        else:
            precision = 0.0
        if self.positive_count > 0:
            recall = self.true_pos_count / self.positive_count
        else:
            recall = 0.0

        if precision + recall > 0:
            self.value = 2 * precision * recall / (precision + recall)
        else:
            self.value = 0.0

