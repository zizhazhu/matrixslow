import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder


class Trainer:

    def __init__(self, x, y, predict, optimizer, batch_size=64):
        self.optimizer = optimizer
        self.x = x
        self.y = y
        self.predict = predict
        self.batch_size = batch_size

    def train(self, features, labels, n_epochs=10, is_one_hot=False):
        if is_one_hot is False:
            one_hot_encoder = OneHotEncoder(sparse=False)
            one_hot_label = one_hot_encoder.fit_transform(labels.reshape(-1, 1))
        else:
            one_hot_label = labels

        for epoch in range(n_epochs):
            batch_count = 0
            for i in tqdm(range(len(features))):
                feature = np.mat(features[i]).T
                label = np.mat(one_hot_label[i]).T

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

            pred = np.array(pred).argmax(axis=1)
            acc = (labels == pred).astype(int).sum() / len(features)
            print(f"Epoch: {epoch}, accuracy: {acc:.3f}")
