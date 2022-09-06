import numpy as np

import matrixslow as ms
from matrixslow.dataset.gender_data import gen_data


class Adaline:

    def __init__(self):
        # 容器变量
        self.x = ms.core.Variable(dim=(3, 1), init=False, trainable=False)
        self.label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

        # parameters
        self.weights = ms.core.Variable(dim=(1, 3), init=True, trainable=True)
        self.bias = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

    def forward(self):
        output = ms.ops.Add(ms.ops.MatMul(self.weights, self.x), self.bias)
        return output


# hyper-parameters
learning_rate = 0.0001
epochs_num = 100
batch_size = 8

train_set = gen_data()

model = Adaline()
output = model.forward()
predict = ms.ops.Step(output)
loss = ms.ops.loss.PerceptionLoss(ms.ops.Multiply(model.label, output))
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

cur_batch_size = 0
for epoch in range(epochs_num):
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        l = np.mat(train_set[i, -1])

        model.x.set_value(features)
        model.label.set_value(l)

        optimizer.one_step()
        cur_batch_size += 1

        if cur_batch_size == batch_size:
            optimizer.update()
            cur_batch_size = 0

    pred = []

    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        model.x.set_value(features)

        predict.forward()
        pred.extend(predict.value.ravel())

    pred = np.array(pred) * 2 - 1
    accuracy = (train_set[:, -1] == pred).astype(int).sum() / len(train_set)
    print(f"epoch: {epoch+1}, accuracy: {accuracy:.3}")


