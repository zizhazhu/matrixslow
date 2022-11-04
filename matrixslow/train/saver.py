import json
import os
from datetime import datetime

import numpy as np

import matrixslow as ms


class Saver:

    def __init__(self, save_path='./model'):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    @staticmethod
    def create_node(graph, model_json, node_json):
        node_type = node_json['node_type']
        inputs_name = node_json['inputs']
        kwargs = node_json['kwargs']

        inputs = []
        for input_name in inputs_name:
            input_node = graph.get_node_by_name(input_name)
            if input_node is None:
                input_node_json = model_json[input_name]
                input_node = Saver.create_node(graph, model_json, input_node_json)
            inputs.append(input_node)

        if node_type == 'Variable':
            return ms.core.Variable(**kwargs)
        else:
            return ClassMining.get_instance_by_subclass_name(ms.core.Node, node_type)(inputs, **kwargs)

    def save(self, graph=ms.core.default_graph, service_signature=None):
        # checkpoint's information
        meta = {
            'save_time': str(datetime.now()),
        }

        service = service_signature
        self._save_model_and_weights(graph, meta, service)

    def _save_model_and_weights(self, graph, meta, service):
        model_json = {
            'meta': meta,
            'service': service,
        }
        graph_json = {}
        weights_dict = {}
        for name, node in graph.nodes.items():
            if not node.need_save:
                continue
            node_json = {
                'node_type': node.__class__.__name__,
                'name': name,
                'inputs': [input_node._name for input_node in node.inputs],
                'outputs': [output_node._name for output_node in node.outputs],
                'kwargs': node.kwargs,
            }
            if node.value is not None:
                node_json['dim'] = node.value.shape
            graph_json[name] = node_json
            if isinstance(node, ms.core.Variable):
                weights_dict[name] = node.value

        model_json['graph'] = graph_json
        model_file_path = os.path.join(self.save_path, 'model.json')
        with open(model_file_path, 'w') as f:
            json.dump(model_json, f, indent=4)

        weights_file_path = os.path.join(self.save_path, 'weights.npy')
        np.savez(weights_file_path, **weights_dict)

    def _restore_node(self, graph, model_json, weights_dict):
        for i in range(len(model_json)):
            node_json = model_json[i]
            node_name = node_json['name']

            weights = None
            if node_name in weights_dict:
                weights = weights_dict[node_name]

            node = graph.get_node_by_name(node_name)
            if node is None:
                print(f'Node {node_name} of type {node_json["node_type"]} not found in graph. Try to create it.')
                node = Saver.create_node(graph, model_json, node_json)
            node.value = weights

    def load(self, graph=ms.core.default_graph):
        weights_dict = {}

        model_file_path = os.path.join(self.save_path, 'model.json')
        with open(model_file_path, 'r') as f:
            model_json = json.load(f)

        weights_file_path = os.path.join(self.save_path, 'weights.npy')
        with open(weights_file_path, 'rb') as f:
            weights_files = np.load(f)
            for filename in weights_files.files:
                weights_dict[filename] = weights_files[filename]
            weights_files.close()

        graph_json = model_json['graph']
        self._restore_node(graph, graph_json, weights_dict)
        print(f'Load model from {self.save_path} successfully.')
        self.meta = model_json.get('meta', None)
        self.service = model_json.get('service', None)
        return self.meta, self.service


class ClassMining:

    @classmethod
    def get_instance_by_subclass_name(cls, model, name):
        for subclass in model.__subclasses__():
            if subclass.__name__ == name:
                return subclass
            instance = cls.get_instance_by_subclass_name(subclass, name)
            if instance:
                return instance
        return None


