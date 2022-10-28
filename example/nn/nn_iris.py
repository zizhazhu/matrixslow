import numpy as np
from sklearn.preprocessing import OneHotEncoder

import matrixslow as ms
from matrixslow.dataset.iris_data import gen_data

features, labels = gen_data()

one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_labels = one_hot_encoder.fit_transform(labels.reshape(-1, 1))

x = ms.core.Variable(dim=(4, 1), init=False, trainable=False)
y = ms.core.Variable(dim=(3, 1), init=False, trainable=False)

model = ms.model.NN(input_size=4, layers=(10, 10), output_size=3)
logits = model.forward(x)
predict = ms.ops.Softmax(logits)

loss = ms.ops.loss.CrossEntropyWithSoftMax(logits, y)

learning_rate = 0.02
batch_size = 8
n_epochs = 50
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

trainer = ms.train.SimpleTrainer(optimizer, metric_ops=[ms.ops.metrics.Accuracy(predict, y),
                                                        ms.ops.metrics.Precision(predict, y)])
trainer.train_and_test(train_dict={x: features, y: one_hot_labels}, test_dict={x: features, y: one_hot_labels},
                       n_epochs=n_epochs)
