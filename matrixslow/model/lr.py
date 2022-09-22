import matrixslow as ms


class LR:

    def __init__(self, dim, quadratic=False):
        self.quadratic = quadratic
        self.dim = dim
        if self.quadratic:
            dimensions = dim * (dim + 1)
            self.w = ms.core.Variable(dim=(1, dimensions), init=True, trainable=True)
        else:
            self.w = ms.core.Variable(dim=(1, dim), init=True, trainable=True)
        self.b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

    def forward(self, x):
        if self.quadratic:
            x_2 = ms.ops.Reshape(
                ms.ops.MatMul(x, ms.ops.Reshape(x, shape=(1, self.dim))),
                shape=(self.dim * self.dim, 1),
            )
            x = ms.ops.Concat([x, x_2])
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


class LRQuadratic:

    def __init__(self, dim):
        self.dim = dim
        self.w = ms.core.Variable(dim=(1, dim), init=True, trainable=True)
        self.W = ms.core.Variable(dim=(dim, dim), init=True, trainable=True)
        self.b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

    def forward(self, x):

        logits = ms.ops.Add(ms.ops.MatMul(self.w, x),
                            ms.ops.MatMul(ms.ops.Reshape(x, shape=(1, self.dim)), ms.ops.MatMul(self.W, x)),
                            self.b,
                            )
        predict = ms.ops.Sigmoid(logits)
        return logits, predict
