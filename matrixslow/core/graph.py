
class Graph:
    def __init__(self):
        self.nodes = {}
        self.name_scope = None

    def clear_jacobi(self):
        for node in self.nodes.values():
            node.clear_jacobi()

    def add_node(self, node, name):
        if name in self.nodes:
            name = f'{node.__class__.__name__}:{self.node_count}'
        self.nodes[name] = node
        return name

    def get_node_by_name(self, name):
        return self.nodes.get(name, None)

    @property
    def node_count(self):
        return len(self.nodes)


default_graph = Graph()
