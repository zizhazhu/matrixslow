from sklearn.preprocessing import OneHotEncoder

import matrixslow as ms
from matrixslow.dataset.mnist import gen_data

features, labels = gen_data()

img_shape = (28, 28)

x = ms.core.Variable(dim=(28, 28), init=False, trainable=False)
y = ms.core.Variable(dim=(10, 1), init=False, trainable=False)

conv1 = ms.layer.conv([x], img_shape, 3, (5, 5), ms.ops.ReLU)
pooling1 = ms.layer.pooling(conv1, (3, 3), (2, 2))
conv2 = ms.layer.conv(pooling1, (14, 14), 3, (3, 3), ms.ops.ReLU)
pooling2 = ms.layer.pooling(conv2, (3, 3), (2, 2))
cnn_output = pooling2
fc1 = ms.layer.dense(ms.ops.Concat(cnn_output, axis=1), 147, 120, ms.ops.ReLU)
output = ms.layer.dense(fc1, 120, 10, None)
predict = ms.ops.Softmax(output)
loss = ms.ops.loss.CrossEntropyWithSoftMax(output, y)

lr = 0.005
batch_size = 64
optimizer = ms.optimizer.Adam(ms.default_graph, loss, lr)

trainer = ms.train.Trainer(x, y, predict, optimizer)
trainer.train_and_test(features, labels, n_epochs=1, one_hot=True)
