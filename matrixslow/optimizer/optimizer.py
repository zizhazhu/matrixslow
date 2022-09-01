class Optimizer:

    def __init__(self, graph, target, learning_rate=0.01):
        self.graph = graph
        self.target = target
        self.learning_rate = learning_rate

        self.acc_gradient = dict()
        self.acc_no = 0
