import numpy as np
import matrixslow as ms
from matrixslow.dataset.trigonometric import get_sequence_data

seq_len = 96
dimension = 16
status_dimension = 12
train_ratio = 0.8

features, labels = get_sequence_data(length=seq_len, dimension=dimension)
train_size = int(len(features) * train_ratio)
train_features, train_labels = features[:train_size], labels[:train_size]
test_features, test_labels = features[train_size:], labels[train_size:]

inputs = [ms.core.Variable(dim=(dimension, 1), init=False, trainable=False, name='x') for _ in range(seq_len)]
U = ms.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)
W = ms.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)
b = ms.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)
hiddens = []

last_step = None
# h = Ux + Wh + b
for iv in inputs:
    h = ms.ops.Add(ms.ops.MatMul(U, iv), b)
    if last_step is not None:
        h = ms.ops.Add(ms.ops.MatMul(W, last_step), h)
    h = ms.ops.ReLU(h)
    last_step = h
    hiddens.append(h)

welding_point = ms.ops.Welding()

fc_model = ms.model.NN(status_dimension, (32, 16), 2)
logits = fc_model.forward(welding_point)
predict = ms.ops.Softmax(logits)

label = ms.core.Variable(dim=(2, 1), init=False, trainable=False, name='label')
loss = ms.ops.loss.CrossEntropyWithSoftMax(logits, label)

learning_rate = 0.005
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

batch_size = 16

for epoch in range(30):

    batch_count = 0
    for i, s in enumerate(train_features):
        start = np.random.randint(len(s) // 3)
        end = np.random.randint(len(s) // 3 + 30, len(s))
        s = s[start: end]

        # 将变长的输入序列赋给RNN的各输入向量节点
        for j in range(len(s)):
            inputs[j].set_value(np.mat(s[j]).T)

        # 将临时的最后一个时刻与全连接网络焊接
        welding_point.weld(hiddens[len(s) - 1])
        label.set_value(np.mat(train_labels[i, :]).T)

        optimizer.one_step()

        batch_count += 1
        if batch_count >= batch_size:
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))
            optimizer.update()
            batch_count = 0

    pred = []
    for i, s in enumerate(test_features):

        start = np.random.randint(len(s) // 3)
        end = np.random.randint(len(s) // 3 + 30, len(s))
        s = s[start: end]

        for j in range(len(s)):
            inputs[j].set_value(np.mat(s[j]).T)

        welding_point.weld(hiddens[len(s) - 1])

        predict.forward()
        pred.append(predict.value.A.ravel())

    pred = np.array(pred).argmax(axis=1)
    true = test_labels.argmax(axis=1)

    accuracy = (true == pred).astype(np.int).sum() / len(test_labels)
    print("epoch: {:d}, accuracy: {:.5f}".format(epoch + 1, accuracy))
