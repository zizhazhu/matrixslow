import matrixslow as ms
from matrixslow.dataset.circle import gen_data

features, labels = gen_data(600, noise_dimension=18)

x = ms.core.Variable(dim=(20, 1), init=False, trainable=False)
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

model = ms.model.FM(dim=20, k=2)
logits, predict = model.forward(x)

loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, logits))

learning_rate = 0.001
batch_size = 16
nepochs = 50

optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

trainer = ms.train.Trainer(x, label, predict, optimizer)
trainer.train(features, labels, n_epochs=nepochs)
