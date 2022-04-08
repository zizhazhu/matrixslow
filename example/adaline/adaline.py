import numpy as np
import matrixslow as ms

male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [-1] * 500

train_data = np.array([np.concatenate((male_heights, female_heights)),
                       np.concatenate((male_weights, female_weights)),
                       np.concatenate((male_bfrs, female_bfrs)),
                       np.concatenate((male_labels, female_labels)),
                       ]).T
np.random.shuffle(train_data)

x = ms.core.Variable(dim=(1, 3), init=False, trainable=False)
y = ms.core.Variable(dim=(1, 1), init=False, trainable=False)
w = ms.core.Variable(dim=(3, 1), init=True, trainable=True)
bias = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

output = ms.ops.Add(ms.ops.MatMul(x, w), bias)

loss = ms.ops.loss.PerceptionLoss(ms.ops.MatMul(y, output))

learning_rate = 0.0001
epochs = 1
for epoch in range(epochs):
    for i in range(len(train_data)):
        features = np.mat(train_data[i,:-1])
        label = np.mat(train_data[i, -1])
        x.set_value(features)
        y.set_value(label)
        loss.forward()

        w.backward(loss)
        bias.backward(loss)