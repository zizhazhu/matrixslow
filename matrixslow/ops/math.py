import numpy as np

import matrixslow as ms
from .operator import Operator


class Add(Operator):
    def __init__(self, *args, name='add'):
        super().__init__(name)
        self.inputs = args
        self.set_output()

    def compute(self):
        shape = self.inputs[0].shape
        self.value = np.mat(np.zeros(shape))
        for input_unit in self.inputs:
            self.value += input_unit.value

    def get_jacobi(self, input_node):
        return np.eye(self.dimension)

    @property
    def kwargs(self):
        return {'name': self._name}
        

class MatMul(Operator):
    def __init__(self, a, b, name='MatMul'):
        super().__init__(name)
        self.inputs = [a, b]
        self.value = None
        self.set_output()

    def compute(self):
        self.value = np.matmul(self.inputs[0].value, self.inputs[1].value)

    def get_jacobi(self, input_node):
        # matrix multiplication jacobi
        zeros = np.zeros((self.dimension, input_node.dimension))
        if input_node is self.inputs[0]:
            return fill_diagonal(zeros, self.inputs[1].value.T)
        else:
            # 调整矩阵顺序，让第一个矩阵的列对齐梯度矩阵的列（这两者维度相同）
            jacobi = fill_diagonal(zeros, self.inputs[0].value)
            row_sort = np.arange(self.dimension).reshape(self.shape[::-1]).T.ravel()
            col_sort = np.arange(input_node.dimension).reshape(input_node.shape[::-1]).T.ravel()
            return jacobi[row_sort, :][:, col_sort]


class Multiply(Operator):
    def __init__(self, a, b, name='Multiply'):
        super().__init__(name=name)
        if not isinstance(b, ms.core.Node):
            b_value = np.mat(b)
            b = ms.core.Variable([1, 1], init=False, trainable=False)
            b.set_value(b_value)
        self.inputs = [a, b]
        self.set_output()

    def compute(self):
        self.value = np.multiply(self.inputs[0].value, self.inputs[1].value)

    def get_jacobi(self, input_node):
        if input_node is self.inputs[0]:
            return np.diag(self.inputs[1].value.A1)
        else:
            return np.diag(self.inputs[0].value.A1)


class Step(Operator):

    def __init__(self, a):
        super().__init__()
        self.inputs = [a]
        self.value = None
        self.set_output()

    def compute(self):
        self.value = np.where(self.inputs[0].value >= 0, 1.0, 0.0)

    def get_jacobi(self, input_node):
        return np.zeros((self.dimension, input_node.dimension))


class Sigmoid(Operator):

    def __init__(self, a):
        super().__init__()
        self.inputs = [a]
        self.set_output()

    def compute(self):
        x = self.inputs[0].value
        self.value = np.mat(1.0 / (1.0 + np.power(np.e, np.where(-x > 1e2, 1e2, -x))))

    def get_jacobi(self, parent):
        # 展开成1维Array再对角化
        return np.diag(np.mat(np.multiply(self.value, 1 - self.value)).A1)


class Softmax(Operator):

    @staticmethod
    def softmax(a):
        a[a > 1e2] = 1e2
        ep = np.power(np.e, a)
        return ep / np.sum(ep)

    def __init__(self, a, name='Softmax'):
        super().__init__(name=name)
        self.inputs = [a]
        self.set_output()

    def compute(self):
        self.value = Softmax.softmax(self.inputs[0].value)

    def get_jacobi(self, input_node):
        raise NotImplementedError("Don't use SoftMax's get_jacobi")


class ReLU(Operator):

    def __init__(self, a, slope=0.1, name='Relu'):
        super().__init__(name=name)
        self.inputs = [a]
        self.slope = slope
        self.set_output()

    def compute(self):
        self.value = np.mat(np.where(self.inputs[0].value >= 0, self.inputs[0].value, self.slope * self.inputs[0].value))

    def get_jacobi(self, input_node):
        return np.diag(np.where(input_node.value.A1 >= 0, 1.0, self.slope))

    @property
    def kwargs(self):
        return {'slope': self.slope, 'name': self._name}


class Square(Operator):

    def __init__(self, a, name='square'):
        super().__init__(name=name)
        self.inputs = [a]
        self.set_output()

    def compute(self):
        self.value = np.square(self.inputs[0].value)

    def get_jacobi(self, input_node):
        return np.diag(self.value * 2)


class Transpose(Operator):
    def __init__(self, node, name='transpose'):
        super().__init__(name)
        self.inputs = [node]
        self.set_output()

    def compute(self):
        self.value = np.transpose(self.inputs[0].value)

    def get_jacobi(self, input_node):
        eye = np.eye(self.dimension)
        row_sort = np.arange(self.dimension).reshape(self.shape[::-1]).T.ravel()
        jacobi = eye[row_sort, :]
        return jacobi


class Welding(Operator):
    def __init__(self, name='welding'):
        super().__init__(name)
        self.inputs = []
        self.set_output()

    def compute(self):
        self.value = self.inputs[0].value

    def get_jacobi(self, input_node):
        return np.mat(np.eye(self.dimension))

    def weld(self, node):

        if len(self.inputs) > 0:
            self.inputs[0].outputs.remove(self)
        self.inputs.clear()
        self.inputs.append(node)
        node.outputs.add(self)


class Convolve(Operator):
    def __init__(self, img, kernel, name='convolve'):
        super().__init__(name)
        self.inputs = [img, kernel]
        self.set_output()
        self.padded = None

    def compute(self):
        img = self.inputs[0].value
        kernel = self.inputs[1].value
        w, h = img.shape
        kw, kh = kernel.shape
        hkw, hkh = kw // 2, kh // 2

        pw, ph = w + kw - 1, h + kh - 1
        self.padded = np.mat(np.zeros((pw, ph)))
        self.padded[hkw:hkw + w, hkh:hkh + h] = img
        self.value = np.mat(np.zeros((w, h)))

        for i in range(hkw, hkw + w):
            for j in range(hkh, hkh + h):
                self.value[i - hkw, j - hkh] = np.sum(np.multiply(
                    self.padded[i - hkw:i + hkw + 1, j - hkh:j + hkh + 1], kernel))

    def get_jacobi(self, input_node):
        img = self.inputs[0].value
        kernel = self.inputs[1].value
        w, h = img.shape
        kw, kh = kernel.shape
        hkw, hkh = kw // 2, kh // 2
        pw, ph = w + kw - 1, h + kh - 1
        jacobi = []
        # 计算有问题
        if input_node is self.inputs[0]:
            for i in range(hkw, hkw + w):
                for j in range(hkh, hkh + h):
                    mask = np.mat(np.zeros((pw, ph)))
                    mask[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh] = kernel
                    jacobi.append(mask[hkw:hkw + w, hkh:hkh + h].A1)
        else:
            for i in range(hkw, hkw + w):
                for j in range(hkh, hkh + h):
                    jacobi.append(
                        self.padded[i - hkw:i - hkw + kw, j - hkh:j - hkh + kh].A1)
        return np.mat(jacobi)


class MaxPooling(Operator):

    def __init__(self, *inputs, size=(2, 2), stride=(1, 1), name='max_pooling'):
        super().__init__(name=name)
        self.inputs = inputs
        self._stride = tuple(stride)
        self._size = tuple(size)
        self.flag = None
        self.set_output()

    def compute(self):
        image = self.inputs[0].value
        w, h = image.shape
        dim = w * h
        sw, sh = self._stride
        kw, kh = self._size
        hkw, hkh = kw // 2, kh // 2

        result = []
        flag = []

        for i in range(0, w, sw):
            row = []
            for j in range(0, h, sh):
                top, bottom = max(0, i - hkw), min(w, i + hkw + 1)
                left, right = max(0, j - hkh), min(h, j + hkh + 1)
                mask = image[top:bottom, left:right]
                row.append(np.max(mask))

                pos = np.argmax(mask)
                w_width = right - left
                offset_w, offset_h = top + pos // w_width, left + pos % w_width
                offset = offset_w * w + offset_h
                tmp = np.zeros(dim)
                tmp[offset] = 1
                flag.append(tmp)

            result.append(row)

        self.value = np.mat(result)
        self.flag = np.mat(flag)

    def get_jacobi(self, input_node):
        return self.flag

    @property
    def kwargs(self):
        return {'size': self._size, 'stride': self._stride, 'name': self._name}


class ScalarMultiply(Operator):
    def __init__(self, scalar, matrix, name='scalar_multiply'):
        super().__init__(name)
        self.inputs = [scalar, matrix]
        self.set_output()

    def compute(self):
        self.value = np.multiply(self.inputs[0].value, self.inputs[1].value)

    def get_jacobi(self, input_node):
        if input_node is self.inputs[0]:
            return self.inputs[1].value.flatten().T
        else:
            return np.mat(np.eye(self.inputs[1].dimension)) * self.inputs[0].value[0, 0]


def fill_diagonal(to_be_filled, filler):
    factor = int(to_be_filled.shape[0] / filler.shape[0])
    m, n = filler.shape
    for i in range(factor):
        to_be_filled[i*m:(i+1)*m, i*n:(i+1)*n] = filler
    return to_be_filled
