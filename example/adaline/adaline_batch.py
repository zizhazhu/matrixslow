import numpy as np

import matrixslow as ms
from matrixslow.dataset.gender_data import gen_data

train_data = gen_data()

batch_size = 10

# 容器变量
x = ms.core.Variable(dim=(batch_size, 3), init=False, trainable=False)
label = ms.core.Variable(dim=(batch_size, 1), init=False, trainable=False)

# parameters
weights = ms.core.Variable(dim=(3, 1), init=True, trainable=True)
bias = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# extend bias
ones = ms.core.Variable(dim=(batch_size, 1), init=False, trainable=False)
ones.set_value(np.mat(np.ones(batch_size)).T)
bias_extend = ms.ops.MatMul(ones, bias)

output = ms.ops.Add(ms.ops.MatMul(x, weights), bias_extend)
predict = ms.ops.Step(output)

loss = ms.ops.loss.PerceptionLoss(ms.ops.Multiply(label, output))

# 将loss压缩到1个
batch_avg = ms.core.Variable(dim=(1, batch_size), init=False, trainable=False)
batch_avg.set_value(1 / batch_size * np.mat(np.ones(batch_size)))
mean_loss = ms.ops.MatMul(batch_avg, loss)

# hyper-parameters
learning_rate = 0.0001
epochs_num = 50

for epoch in range(epochs_num):
    for i in np.arange(0, len(train_data), batch_size):
        features = np.mat(train_data[i:i+batch_size, :-1])
        l = np.mat(train_data[i:i+batch_size, -1]).T

        x.set_value(features)
        label.set_value(l)

        mean_loss.forward()
        weights.backward(mean_loss)
        bias.backward(mean_loss)

        weights.set_value(weights.value - learning_rate * weights.grad.reshape(weights.shape))
        bias.set_value(bias.value - learning_rate * bias.grad.reshape(bias.shape))

        ms.default_graph.clear_jacobi()

    pred = []

    for i in np.arange(0, len(train_data), batch_size):
        features = np.mat(train_data[i:i+batch_size, :-1])
        x.set_value(features)

        predict.forward()
        pred.extend(predict.value.ravel())

    pred = np.array(pred) * 2 - 1
    accuracy = (train_data[:, -1] == pred).astype(np.int).sum() / len(train_data)
    print(f"epoch: {epoch+1}, accuracy: {accuracy:.3}")


