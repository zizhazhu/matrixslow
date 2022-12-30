import numpy as np

from ..core import Node
from .proto import common_pb2


class DistCommon:

    @staticmethod
    def _serialize_proto_node_gradients(node_gradients_dict):
        proto_node_gradients = common_pb2.NodeGradients()
        for node_name, gradients in node_gradients_dict.items():
            proto_node = proto_node_gradients.nodes.add()
            if isinstance(node_name, Node):
                proto_node.name = node_name.name
            else:
                proto_node.name = node_name
            proto_gradient = proto_node_gradients.gradients.add()
            proto_gradient.value.extend(np.array(gradients).flatten())
            proto_gradient.dim.extend(list(gradients.shape))

        return proto_node_gradients

    @staticmethod
    def _deserialize_proto_node_gradients(node_gradients):
        proto_nodes = node_gradients.nodes
        proto_gradients = node_gradients.gradients

        assert len(proto_nodes) == len(proto_gradients)

        node_with_gradients = dict()

        for index in range(len(proto_nodes)):
            node_name = proto_nodes[index].name
            gradients_value = proto_gradients[index].value
            gradients_dim = tuple(proto_gradients[index].dim)
            gradient_mat = np.mat(gradients_value, dtype=np.float32)
            gradient_mat = np.reshape(gradient_mat, gradients_dim)
            node_with_gradients[node_name] = gradient_mat
        return node_with_gradients

    @staticmethod
    def _serialize_proto_variable_weights(varibale_weights_dict):
        var_weights_req_resp = common_pb2.VariableWeightsReqResp()
        for name, mat in varibale_weights_dict.items():
            var = var_weights_req_resp.nodes.add()
            if isinstance(name, Node):
                name = name.name
            var.name = name
            weight = var_weights_req_resp.weights.add()

            weight.value.extend(np.array(mat).flatten())
            weight.dim.extend(list(mat.shape))

        return var_weights_req_resp

    @staticmethod
    def _deserialize_proto_variable_weights(variable_weights_req_resp):
        proto_variables = variable_weights_req_resp.nodes
        proto_weights = variable_weights_req_resp.weights

        assert len(proto_variables) == len(proto_weights)

        var_weights_dict = dict()

        for index in range(len(proto_variables)):
            var_name = proto_variables[index].name
            weights_value = proto_weights[index].value
            weigths_dim = tuple(proto_weights[index].dim)
            weights_mat = np.mat(weights_value, dtype=np.float32)
            weights_mat = np.reshape(weights_mat, weigths_dim)
            var_weights_dict[var_name] = weights_mat

        return var_weights_dict
