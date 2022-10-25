import abc

import numpy as np
from tqdm import tqdm


class Trainer:

    def __init__(self, x, y, predict, optimizer, batch_size=64, is_multi_class=False):
        self.optimizer = optimizer
        self.x = x
        self.y = y
        self.predict = predict
        self.batch_size = batch_size
        self._is_multi_class = is_multi_class

    def train_and_test(self, train_x, train_y, test_x=None, test_y=None, n_epochs=10):
        self._variable_weights_init()
        self.main_loop(train_x, train_y, test_x, test_y, n_epochs)

    def main_loop(self, train_x, train_y, test_x=None, test_y=None, n_epochs=10):

        if test_x is None or test_y is None:
            test_x = train_x
            test_y = train_y

        for epoch in range(n_epochs):
            batch_count = 0
            for i in tqdm(range(len(train_x))):
                if isinstance(self.x, list):
                    for j, x in enumerate(self.x):
                        feature = np.mat(train_x[i][j]).T
                        x.set_value(feature)
                else:
                    feature = np.mat(train_x[i]).T
                    self.x.set_value(feature.reshape(self.x.dim))

                label = np.mat(train_label[i]).T
                self.y.set_value(label)

                self.optimizer.one_step()
                batch_count += 1

                if batch_count >= self.batch_size:
                    self.optimizer.update()
                    batch_count = 0

            pred = []
            for i in range(len(test_x)):
                if isinstance(self.x, list):
                    for j, x in enumerate(self.x):
                        feature = np.mat(test_x[i][j]).T
                        x.set_value(feature)
                else:
                    feature = np.mat(test_x[i]).T
                    self.x.set_value(feature.reshape(self.x.dim))
                self.predict.forward()
                pred.append(self.predict.value.A.ravel())

            if self._is_multi_class:
                pred = np.array(pred).argmax(axis=1)
            else:
                pred = (np.array(pred) > 0.5).astype(int) * 2 - 1
                pred = np.squeeze(pred)
            acc = (test_y == pred).astype(int).sum() / len(test_x)
            print(f"Epoch: {epoch}, accuracy: {acc:.3f}")

    def train(self):
        pass

    def eval(self):
        pass

    def _get_input_values(self, x):
        pass

    def one_step(self):
        pass

    @abc.abstractmethod
    def _variable_weights_init(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def _optimizer_update(self):
        raise NotImplementedError()

