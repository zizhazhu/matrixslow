import numpy as np
from sklearn.preprocessing import OneHotEncoder

import matrixslow as ms
from matrixslow.dataset.iris_data import gen_data

features, labels = gen_data()

one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_label = one_hot_encoder.fit_transform(labels.reshape(-1, 1))

x = ms.core.Variable(dim=(4, 1), init=False, trainable=False)
y = ms.core.Variable(dim=(3, 1), init=False, trainable=False)

model = ms.model.NN(input_size=4, layers=(10, 10), output_size=3)
logits = model.forward(x)
predict = ms.ops.Softmax(logits)

loss = ms.ops.loss.CrossEntropyWithSoftMax(logits, y)

learning_rate = 0.002
batch_size = 8
n_epochs = 30
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

for epoch in range(n_epochs):
    batch_count = 0
    for i in range(len(features)):
        feature = np.mat(features[i]).T
        label = np.mat(one_hot_label[i]).T

        x.set_value(feature)
        y.set_value(label)

        optimizer.one_step()
        batch_count += 1

        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0

    pred = []
    for i in range(len(features)):
        feature = np.mat(features[i]).T
        x.set_value(feature)
        predict.forward()
        pred.append(predict.value.A.ravel())

    pred = np.array(pred).argmax(axis=1)
    acc = (labels == pred).astype(int).sum() / len(features)
    print(f"Epoch: {epoch}, accuracy: {acc:.3f}")


