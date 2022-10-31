# TODO: finish
import matrixslow as ms
from matrixslow.dataset.titanic import gen_data

features, labels = gen_data(file='./data/titanic.csv')

dimension = features.shape[1]
x = ms.core.Variable(dim=(dimension, 1), init=False, trainable=False)
y = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

model = ms.model.WideAndDeep(dimension, embedding_size=16, layers=(8, 4))
logits = model.forward(x)
predict = ms.ops.Sigmoid(logits)

loss = ms.ops.loss.LogLoss(ms.ops.Multiply(logits, y))

learning_rate = 0.005
batch_size = 16
n_epochs = 50
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

trainer = ms.train.SimpleTrainer(optimizer, metric_ops=[ms.ops.metrics.Accuracy(predict, y),
                                                        ms.ops.metrics.Precision(predict, y),
                                                        ms.ops.metrics.AUC(predict, y),
                                                        ])
trainer.train_and_test(train_dict={x: features, y: labels}, test_dict={x: features, y: labels},
                       n_epochs=n_epochs)
