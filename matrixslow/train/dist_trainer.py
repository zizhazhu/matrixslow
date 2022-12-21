from .trainer import Trainer
from ..dist import ps

import matrixslow as ms


class DistPSTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        cluster_conf = kwargs['cluster_conf']
        ps_host = cluster_conf['ps'][0]
        self.ps_client = ps.ParameterServiceClient(ps_host)

    def _variable_weights_init(self):
        var_weights_dict = dict()
        for node in ms.core.default_graph.nodes:
            if isinstance(node, ms.core.Variable) and node.trainable:
                var_weights_dict[node.name] = node.value

        duplicated_var_weights_dict = self.ps_client.variable_weights_init(var_weights_dict)
        for name, weights in duplicated_var_weights_dict.items():
            ms.core.update_node_value_in_graph(name, weights)

        print('[DistPSTrainer] Variable weights init done.')

    def _optimizer_update(self):
        acc_gradients = self.optimizer.acc_gradients
        self.ps_client.push_gradients(acc_gradients, self.optimizer.acc_no)
        node_gradients_dict = self.ps_client.pull_gradients()
        self.optimizer.update(node_gradients_dict)
