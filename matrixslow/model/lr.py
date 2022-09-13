import matrixslow as ms


class LR:

    def __init__(self):
        self.w = ms.core.Variable(dim=(1, 3), init=True, trainable=True)
        self.b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

    def forward(self, x):
        logits = ms.ops.Add(ms.ops.MatMul(self.w, x), self.b)
        predict = ms.ops.Sigmoid(logits)
        return logits, predict


class MultiLabelLR:

    def __init__(self, input_dim, output_dim):
        self.w = ms.core.Variable(dim=(output_dim, input_dim), init=True, trainable=True)
        self.b = ms.core.Variable(dim=(output_dim, 1), init=True, trainable=True)

    def forward(self, x):
        logits = ms.ops.Add(ms.ops.MatMul(self.w, x), self.b)
        predict = ms.ops.Softmax(logits)
        return logits, predict
