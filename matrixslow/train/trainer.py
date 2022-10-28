import abc

import numpy as np
from tqdm import tqdm


class Trainer:

    def __init__(self, optimizer, batch_size=64, metric_ops=None):
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.metric_ops = list(metric_ops)

    def train_and_test(self, train_dict, test_dict=None, n_epochs=10):
        self._variable_weights_init()
        self.main_loop(train_dict, test_dict, n_epochs)

    def main_loop(self, train_dict, test_dict=None, n_epochs=10):

        for epoch in range(n_epochs):
            self.train(train_dict)
            if test_dict is not None:
                self.eval(test_dict)

    def train(self, train_dict):
        for i in tqdm(range(len(list(train_dict.values())[0]))):
            self._set_input_values(train_dict, i)
            self.optimizer.one_step()

            if (i + 1) % self.batch_size == 0:
                self._optimizer_update()

    def eval(self, test_dict):
        for metric_op in self.metric_ops:
            metric_op.reset()
        for i in tqdm(range(len(list(test_dict.values())[0]))):
            self._set_input_values(test_dict, i)

            for metric_op in self.metric_ops:
                metric_op.forward()

        metrics_str = 'Evaluation metrics '
        for metric_op in self.metric_ops:
            metrics_str += metric_op.value_str()
        print(metrics_str)

    def _set_input_values(self, feed_dict, index):
        for node, value in feed_dict.items():
            node.set_value(np.mat(value[index]).T)

    @abc.abstractmethod
    def _variable_weights_init(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _optimizer_update(self):
        raise NotImplementedError()

