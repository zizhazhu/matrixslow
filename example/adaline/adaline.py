import numpy as np
import matrixslow as ms

from matrixslow.dataset.gender_data import gen_data

train_data = gen_data()

# 采用和batch模式一致的维度定义，每一行是一条样本
x = ms.core.Variable(dim=(1, 3), init=False, trainable=False)
y = ms.core.Variable(dim=(1, 1), init=False, trainable=False)
w = ms.core.Variable(dim=(3, 1), init=True, trainable=True)
bias = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# 按照batch模式排列，所以是XW+b
output = ms.ops.Add(ms.ops.MatMul(x, w), bias)
predict = ms.ops.Step(output)

loss = ms.ops.loss.PerceptionLoss(ms.ops.MatMul(y, output))

learning_rate = 0.0001
epochs = 10
for epoch in range(epochs):
    for i in range(len(train_data)):
        features = np.mat(train_data[i,:-1])
        label = np.mat(train_data[i, -1])
        x.set_value(features)
        y.set_value(label)
        loss.forward()

        w.backward(loss)
        bias.backward(loss)

        w.set_value(w.value - learning_rate * w.grad.reshape(w.shape))
        bias.set_value(bias.value - learning_rate * bias.grad.reshape(bias.shape))

        ms.default_graph.clear_jacobi()

    pred = []

    for i in range(len(train_data)):
        features = np.mat(train_data[i,:-1])
        x.set_value(features)
        predict.forward()
        pred.append(predict.value[0, 0])

    pred = np.array(pred) * 2 - 1
    accuracy = (train_data[:, -1] == pred).mean()
    print(f'Epoch {epoch}: {accuracy}')
