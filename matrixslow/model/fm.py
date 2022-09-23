import matrixslow as ms


class FM:

    def __init__(self, dim, k=8):
        self._dim = dim
        self._k = k

        self._w = ms.core.Variable(dim=(1, dim), init=True, trainable=True, name='w')
        self._h = ms.core.Variable(dim=(k, dim), init=True, trainable=True, name='h')
        self._b = ms.core.Variable(dim=(1, 1), init=True, trainable=True, name='b')

    def forward(self, x):
        hth = ms.ops.MatMul(ms.ops.Transpose(self._h, name='h_t'), self._h, name='hth')
        logits = ms.ops.Add(
            ms.ops.MatMul(self._w, x, name='wx'),
            ms.ops.MatMul(ms.ops.Transpose(x, name='xt'),
                          ms.ops.MatMul(hth, x, name='hthx'), name='xhhx'),
            self._b,
        )
        predict = ms.ops.Sigmoid(logits)
        return logits, predict


