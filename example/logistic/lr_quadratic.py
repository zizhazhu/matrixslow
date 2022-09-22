import numpy as np

import matrixslow as ms
from matrixslow.dataset.circle import gen_data

features, labels = gen_data()

x = ms.core.Variable(dim=(2, 1), init=False, trainable=False)
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

old = False
if old:
    # logits = w((x.T x).flatten concat x) + b
    model = ms.model.LR(dim=2, quadratic=True)
else:
    # logits = x.T W x + wx + b
    model = ms.model.LRQuadratic(dim=2)
logits, predict = model.forward(x)

loss = ms.ops.loss.LogLoss(ms.ops.Multiply(label, logits))

learning_rate = 0.001
batch_size = 8
nepochs = 200

optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

trainer = ms.train.Trainer(x, label, predict, optimizer)
trainer.train(features, labels)
