import matrixslow as ms
from ..ops import ReLU


def dense(input_layer, input_size, size, activation=ReLU):
    weights = ms.core.Variable(dim=(size, input_size), trainable=True, init=True)
    bias = ms.core.Variable(dim=(size, 1), trainable=True, init=True)
    affine = ms.ops.Add(ms.ops.MatMul(weights, input_layer), bias)
    value = activation(affine)
    return value
