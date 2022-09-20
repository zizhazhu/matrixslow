import numpy as np

import matrixslow as ms
from matrixslow.dataset.circle import gen_data

features, labels = gen_data()

x_1 = ms.core.Variable(dim=(2, 1), init=False, trainable=False)
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

quadratic = False
if quadratic:
    w = ms.core.Variable(dim=(1, 6), init=True, trainable=True)

model = ms.model.LR()
logits, predict = model.forward(x)

loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, logits))

learning_rate = 0.0001
batch_size = 16
nepochs = 50

optimizer = ms.optimizer.GradientDescentMomentum(ms.default_graph, loss, learning_rate)


for epoch in range(nepochs):
    batch_count = 0
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        l = np.mat(train_set[i, -1])
        x.set_value(features)
        label.set_value(l)
        optimizer.one_step()
        batch_count += 1

        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0

    pred = []
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T

        x.set_value(features)
        predict.forward()
        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(int) * 2 - 1
    acc = (train_set[:, -1] == pred).astype(int).sum() / len(train_set)
    print(f"Epoch: {epoch}, accuracy: {acc:.3f}")
