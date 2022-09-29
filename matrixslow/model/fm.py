import numpy as np

import matrixslow as ms


class FM:

    def __init__(self, dim, k=8):
        self._dim = dim
        self._k = k

        self._w = ms.core.Variable(dim=(1, dim), init=True, trainable=True, name='w')
        self._h = ms.core.Variable(dim=(k, dim), init=True, trainable=True, name='h')
        self._b = ms.core.Variable(dim=(1, 1), init=True, trainable=True, name='b')
        self.t1 = ms.core.Variable(dim=(dim, 1), init=False, trainable=False)
        self.t1.set_value(np.ones(shape=(dim, 1)))
        self.t2 = ms.core.Variable(dim=(1, k), init=False, trainable=False)
        self.t2.set_value((np.ones(shape=(1, k))))

    def forward(self, x):
        hx = ms.ops.Multiply(self._h, ms.ops.Transpose(x), name='hx')
        hx_sum = ms.ops.MatMul(hx, self.t1, name='reduced_hx')
        hx_sum_2 = ms.ops.Square(hx_sum, name='hx_sum_2')
        hx_2 = ms.ops.Square(hx, name='hx_2')
        hx_2_sum = ms.ops.MatMul(hx_2, self.t1, name='reduced_hx_2')
        cross = ms.ops.Multiply(ms.ops.Add(hx_sum_2, ms.ops.Multiply(hx_2_sum, -1)), 0.5)
        cross_sum = ms.ops.MatMul(self.t2, cross, name='cross_sum')
        logits = ms.ops.Add(
            ms.ops.MatMul(self._w, x, name='wx'),
            cross_sum,
            self._b,
        )
        predict = ms.ops.Sigmoid(logits)
        return logits, predict


