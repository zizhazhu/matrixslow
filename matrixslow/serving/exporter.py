import matrixslow as ms


class Exporter:

    def __init__(self, graph=ms.core.default_graph):
        self.graph = graph

    def signature(self, input_name, output_name):
        input_node = self.graph.get_node_by_name(input_name)
        output_node = self.graph.get_node_by_name(output_name)
        input_signature = {
            'name': input_name,
        }
        output_signature = {
            'name': output_name,
        }
        return {
            'input': input_signature,
            'output': output_signature,
        }
