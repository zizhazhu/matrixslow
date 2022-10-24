import numpy as np

import matrixslow as ms
from ..ops import ReLU


def dense(input_layer, input_size, size, activation=ReLU):
    weights = ms.core.Variable(dim=(size, input_size), trainable=True, init=True)
    bias = ms.core.Variable(dim=(size, 1), trainable=True, init=True)
    affine = ms.ops.Add(ms.ops.MatMul(weights, input_layer), bias)
    value = activation(affine)
    return value


def conv(feature_maps, input_shape, kernel_size, kernel_shape, activation=ReLU):
    ones = ms.core.Variable(dim=input_shape, trainable=True, init=True)
    ones.set_value(np.mat(np.ones(input_shape)))
    outputs = []
    for i in range(kernel_size):
        channels = []
        for fm in feature_maps:
            # every layer has a kernel
            kernel = ms.core.Variable(kernel_shape, init=True, trainable=True)
            conv = ms.ops.Convolve(fm, kernel)
            channels.append(conv)
        channel_sum = ms.ops.Add(*channels)
        bias = ms.ops.ScalarMultiply(ms.core.Variable((1, 1), init=True, trainable=True), ones)
        affine = ms.ops.Add(channel_sum, bias)
        outputs.append(activation(affine))
    return outputs
