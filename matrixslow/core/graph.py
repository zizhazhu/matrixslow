
class Graph:
    def __init__(self):
        self.nodes = []
        self.name_scope = None

    def clear_jacobi(self):
        for node in self.nodes:
            node.clear_jacobi()

    def add_node(self, node):
        self.nodes.append(node)

    def get_node_by_name(self, name):
        for node in self.nodes:
            if node._name == name:
                return node


default_graph = Graph()