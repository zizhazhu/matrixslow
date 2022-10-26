import abc

import numpy as np
from tqdm import tqdm


class Trainer:

    def __init__(self, x, y, predict, optimizer, batch_size=64, metric_ops=None):
        self.optimizer = optimizer
        self.x = x
        self.y = y
        self.predict = predict
        self.batch_size = batch_size
        self.metric_ops = metric_ops

    def train_and_test(self, train_x, train_y, test_x=None, test_y=None, n_epochs=10):
        self._variable_weights_init()
        self.main_loop(train_x, train_y, test_x, test_y, n_epochs)

    def main_loop(self, train_x, train_y, test_x=None, test_y=None, n_epochs=10):

        for epoch in range(n_epochs):
            self.train(train_x, train_y)
            if test_x is not None:
                self.eval(test_x, test_y)

    def train(self, train_x, train_y):
        for i in tqdm(range(len(train_x.values()[0]))):
            self.one_step(self._get_input_values(train_x, i), train_y[i], is_training=True)

            if (i + 1) % self.batch_size == 0:
                self._optimizer_update()

    def eval(self, test_x, test_y):
        for metric_op in self.metric_ops:
            metric_op.reset()
        for i in tqdm(range(len(test_x.values()[0]))):
            self.one_step(self._get_input_values(test_x, i), test_y[i])

            for metric_op in self.metric_ops:
                metric_op.forward()

        metrics_str = 'Evaluation metrics '
        for metric_op in self.metric_ops:
            metrics_str += metric_op.value_str()
        print(metrics_str)

    def _get_input_values(self, x, index):
        input_values = dict()
        for node_name in x.keys():
            input_values[node_name] = x[node_name][index]
        return input_values

    def one_step(self, data_x, data_y, is_training=False):
        for i in range(len(self.x)):
            input_value = data_x.get(self.x[i].name)
            self.x[i].set_value(np.mat(input_value).T)
        self.y.set_value(np.mat(data_y).T)
        if is_training:
            self.optimizer.one_step()

    @abc.abstractmethod
    def _variable_weights_init(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _optimizer_update(self):
        raise NotImplementedError()

