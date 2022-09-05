import abc

import numpy as np

from ..core.variable import Variable

class Optimizer:

    def __init__(self, graph, target, learning_rate=0.01):
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate

        self.acc_gradient = dict()
        self.acc_no = 0

    def one_step(self):
        self.forward_backward()
        self.acc_no += 1

    def forward_backward(self):
        # 清理梯度，再累加每个节点的梯度
        self.graph.clear_jacobi()
        self.target.forward()

        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)
                # Prob: 为什么要转置，在矩阵有1维是1时没有区别
                gradient = node.grad.T.reshape(node.shape)
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    # 每次节点自己保留的梯度清零，需要累积到优化器内部
                    self.acc_gradient[node] += gradient

    def update(self, var_gradients=None):
        if var_gradients is not None:
            self.apply_gradients(var_gradients)
        self._update()
        # 清理optimizer内部记录的梯度
        self.acc_gradient.clear()
        self.acc_no = 0

    @abc.abstractmethod
    def _update(self):
        # 抽象方法，由每个优化器自己完成
        raise NotImplementedError()

    def apply_gradients(self, node_gradients_dict, summarize=False, acc_no=None):
        # TODO: 暂时用不上，先不实现
        pass

    def get_gradient(self, node):
        return self.acc_gradient[node] / self.acc_no


class GradientDescent(Optimizer):

    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

    def _update(self):
        for node, gradient in self.acc_gradient.items():
            gradient_apply = self.learning_rate * gradient / self.acc_no
            node.set_value(node.value - gradient_apply)


class GradientDescentMomentum(Optimizer):

    def __init__(self, graph, target, learning_rate=0.01, momentum=0.9):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        self.momentum = momentum
        # history momentum
        self.v = dict()

    def _update(self):
        for node, gradient in self.acc_gradient.items():
            gradient_apply = self.learning_rate * gradient / self.acc_no
            if node not in self.v:
                self.v[node] = gradient_apply
            else:
                # v = m * v + \eta * g
                self.v[node] = self.momentum * self.v[node] + gradient_apply
            # w = w - v
            node.set_value(node.value - self.v[node])


class AdaGrad(Optimizer):

    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        # history second-order momentum
        self.s = {}

    def _update(self):
        for node, gradient in self.acc_gradient.items():
            gradient_apply = gradient / self.acc_no

            if node not in self.s:
                self.s[node] = np.power(gradient_apply, 2)
            else:
                self.s[node] += np.power(gradient_apply, 2)

            node.set_value(node.value - self.learning_rate * gradient_apply / (1e-10 + np.sqrt(self.s[node])))
