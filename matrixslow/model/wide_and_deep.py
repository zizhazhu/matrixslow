import matrixslow as ms
from .nn import Layer


class WideAndDeep:

    def __init__(self, feature_size, embedding_size=128, layers=(64, 32), activation=ms.ops.ReLU):
        self._w = ms.core.Variable(dim=(1, feature_size), init=True, trainable=True)
        self._embedding = ms.core.Variable(dim=(embedding_size, feature_size), init=True, trainable=True)
        self.layers = []
        self.layers.append(Layer(embedding_size, layers[0], activation=activation))
        for index in range(len(layers) - 1):
            self.layers.append(Layer(layers[index], layers[index+1], activation=activation))
        self.layers.append(Layer(layers[-1], 1, activation=None))
        self._bias = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

    def forward(self, x):
        wide = ms.ops.MatMul(self._w, x, name='wide')
        # sum reduce
        embedding = ms.ops.MatMul(self._embedding, x, name='embedding')
        out_layer = embedding
        for layer in self.layers:
            out_layer = layer.forward(out_layer)
        result = ms.ops.Add(wide, out_layer, self._bias, name='result')
        return result


