import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


class Trainer:

    def __init__(self, x, y, predict, optimizer, batch_size=64, is_multi_class=False):
        self.optimizer = optimizer
        self.x = x
        self.y = y
        self.predict = predict
        self.batch_size = batch_size
        self._is_multi_class = is_multi_class

    def train(self, features, labels, n_epochs=10, one_hot=False):
        if one_hot is True:
            one_hot_encoder = OneHotEncoder(sparse=False)
            train_label = one_hot_encoder.fit_transform(labels.reshape(-1, 1))
        else:
            train_label = labels

        for epoch in range(n_epochs):
            batch_count = 0
            for i in tqdm(range(len(features))):
                feature = np.mat(features[i]).T
                label = np.mat(train_label[i]).T

                self.x.set_value(feature)
                self.y.set_value(label)

                self.optimizer.one_step()
                batch_count += 1

                if batch_count >= self.batch_size:
                    self.optimizer.update()
                    batch_count = 0

            pred = []
            for i in range(len(features)):
                feature = np.mat(features[i]).T
                self.x.set_value(feature)
                self.predict.forward()
                pred.append(self.predict.value.A.ravel())

            if self._is_multi_class:
                pred = np.array(pred).argmax(axis=1)
            else:
                pred = (np.array(pred) > 0.5).astype(int) * 2 - 1
                pred = np.squeeze(pred)
            acc = (labels == pred).astype(int).sum() / len(features)
            print(f"Epoch: {epoch}, accuracy: {acc:.3f}")
