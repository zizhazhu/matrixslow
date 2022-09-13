import numpy as np
from sklearn.preprocessing import OneHotEncoder

import matrixslow as ms
from matrixslow.dataset.iris_data import gen_data

features, labels = gen_data()
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_label = one_hot_encoder.fit_transform(labels.reshape(-1, 1))

x = ms.core.Variable(dim=(4, 1), init=False, trainable=False)
label = ms.core.Variable(dim=(3, 1), init=False, trainable=False)

model = ms.model.MultiLabelLR(4, 3)
logits, predict = model.forward(x)

loss = ms.ops.loss.CrossEntropyWithSoftMax(logits, label)

learning_rate = 0.005
batch_size = 16
nepochs = 100

optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)


for epoch in range(nepochs):
    batch_count = 0
    for i in range(len(features)):
        feature = np.mat(features[i, :]).T
        l = np.mat(one_hot_label[i, :]).T
        x.set_value(feature)
        label.set_value(l)
        optimizer.one_step()
        batch_count += 1

        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0

    pred = []
    for i in range(len(features)):
        feature = np.mat(features[i, :]).T
        x.set_value(feature)
        predict.forward()
        pred.append(predict.value.A.ravel())

    pred = np.array(pred).argmax(axis=1)
    acc = (labels == pred).astype(int).sum() / len(features)
    print(f"Epoch: {epoch}, accuracy: {acc:.3f}")
