import matrixslow as ms


class NN:

    def __init__(self, input_size, layers=(32,), output_size=1, activation=ms.ops.ReLU):
        self.layers = []
        self.layers.append(Layer(input_size, layers[0], activation=activation))
        for index in range(len(layers) - 1):
            self.layers.append(Layer(layers[index], layers[index+1], activation=activation))
        self.layers.append(Layer(layers[-1], output_size, activation=None))

    def forward(self, x):
        out_layer = x
        for layer in self.layers:
            out_layer = layer.forward(out_layer)
        return out_layer


class Layer:

    def __init__(self, input_size, output_size, activation=ms.ops.ReLU):
        self.w = ms.core.Variable(dim=(output_size, input_size), init=True, trainable=True)
        self.b = ms.core.Variable(dim=(output_size, 1), init=True, trainable=True)
        self.activation = activation

    def forward(self, input_layer):
        affine = ms.ops.Add(ms.ops.MatMul(self.w, input_layer), self.b)
        if self.activation is not None:
            return self.activation(affine)
        else:
            return affine

