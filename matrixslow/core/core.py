from .graph import default_graph


def get_node_from_graph(node_name, name_scope=None, graph=default_graph):
    if name_scope is not None:
        node_name = name_scope + '/' + node_name
    for node in graph.nodes:
        if node.name == node_name:
            return node


def update_node_value_in_graph(node_name, new_value, name_scope=None, graph=default_graph):
    node = get_node_from_graph(node_name, name_scope, graph)
    node.value = new_value
