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

        for name, node in self.graph.nodes.items():
            # 反向传播遍历可训练的节点
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


class RMSProp(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01, beta=0.9):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        self.beta = beta
        self.s = {}

    def _update(self):
        for node, gradient in self.acc_gradient.items():
            gradient_apply = gradient / self.acc_no

            # s = \beta * s + (1 - \beta) * g^2
            if node not in self.s:
                self.s[node] = (1 - self.beta) * np.power(gradient_apply, 2)
            else:
                self.s[node] = self.beta * self.s[node] + (1 - self.beta) * np.power(gradient_apply, 2)

            node.set_value(node.value - self.learning_rate * gradient_apply / (1e-10 + np.sqrt(self.s[node])))


class Adam(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01, beta_1=0.9, beta_2=0.99):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m = {}
        self.v = {}
        self.step = 0

    def _update(self):
        for node, gradient in self.acc_gradient.items():
            gradient_apply = gradient / self.acc_no

            # m = \beta_1 * m + (1 - \beta_1) * g
            # v = \beta_2 * s + (1 - \beta_2) * g^2
            if node not in self.m:
                self.m[node] = (1 - self.beta_1) * gradient_apply
                self.v[node] = (1 - self.beta_2) * np.power(gradient_apply, 2)
            else:
                self.m[node] = self.beta_1 * self.m[node] + (1 - self.beta_1) * gradient_apply
                self.v[node] = self.beta_2 * self.v[node] + (1 - self.beta_2) * np.power(gradient_apply, 2)
            self.step += 1

            # 平滑初期的值
            if self.step > 50:
                m_ = self.m[node]
                v_ = self.v[node]
            else:
                m_ = self.m[node] / (1 - np.power(self.beta_1, self.step))
                v_ = self.v[node] / (1 - np.power(self.beta_2, self.step))
            node.set_value(node.value - self.learning_rate * m_ / (1e-10 + np.sqrt(v_)))
