import matrixslow as ms
from matrixslow.dataset.circle import gen_data

dim = 2
features, labels = gen_data(600, noise_dimension=dim-2)

x = ms.core.Variable(dim=(dim, 1), init=False, trainable=False, name='x')
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False, name='label')

model = ms.model.FM(dim=dim, k=3)
logits, predict = model.forward(x)

loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, logits))

learning_rate = 0.0001
batch_size = 256
nepochs = 50

optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

trainer = ms.train.Trainer(x, label, predict, optimizer)
trainer.train_and_test(features, labels, n_epochs=nepochs)
