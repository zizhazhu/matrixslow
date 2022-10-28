import numpy as np
from sklearn.preprocessing import OneHotEncoder

import matrixslow as ms
from matrixslow.dataset.mnist import gen_data

features, labels = gen_data()

one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_label = one_hot_encoder.fit_transform(labels.reshape(-1, 1))

x = ms.core.Variable(dim=(784, 1), init=False, trainable=False)
y = ms.core.Variable(dim=(10, 1), init=False, trainable=False)

model = ms.model.NN(input_size=784, layers=(100,), output_size=10)
logits = model.forward(x)
predict = ms.ops.Softmax(logits)

loss = ms.ops.loss.CrossEntropyWithSoftMax(logits, y)

learning_rate = 0.001
batch_size = 64
n_epochs = 30
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

trainer = ms.train.Trainer(optimizer, batch_size=batch_size, metric_ops=[ms.ops.metrics.Accuracy(predict, y)])
trainer.train_and_test({x: features, y: one_hot_label})
