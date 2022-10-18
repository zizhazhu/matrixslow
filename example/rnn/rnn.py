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

last_step = None
# h = Ux + Wh + b
for iv in inputs:
    h = ms.ops.Add(ms.ops.MatMul(U, iv), b)
    if last_step is not None:
        h = ms.ops.Add(ms.ops.MatMul(W, last_step), h)
    h = ms.ops.ReLU(h)
    last_step = h

fc_model = ms.model.NN(status_dimension, (32, 16), 2)
logits = fc_model.forward(last_step)
predict = ms.ops.Softmax(logits)

label = ms.core.Variable(dim=(2, 1), init=False, trainable=False, name='label')
loss = ms.ops.loss.CrossEntropyWithSoftMax(logits, label)

learning_rate = 0.005
optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

batch_size = 16

trainer = ms.train.Trainer(inputs, label, predict, optimizer, batch_size=batch_size)
trainer.train_and_test(train_features, train_labels, test_features, test_labels, n_epochs=50)
