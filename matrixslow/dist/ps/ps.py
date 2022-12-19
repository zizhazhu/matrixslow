import time
import threading
from concurrent.futures import ThreadPoolExecutor

import grpc

from ..dist import DistCommon
from ..proto import parameter_server_pb2, parameter_server_pb2_grpc


class ParameterServer(parameter_server_pb2_grpc.ParameterServiceServicer):

    def __init__(self, worker_num, sync=True):
        self.worker_num = worker_num
        self.cur_pull_num = self.worker_num
        self.cond = threading.Condition()

    def Push(self, request, context):
        node_with_gradients, acc_no = self._deserialize_push_req(request)

    def _push_sync(self, node_with_gradients, acc_no):
        if self.cond.acquire():
            while self.cur_pull_num != self.worker_num:
                self.cond.wait()


    def _deserialize_push_req(self, request):
        acc_no = request.node_gradients.acc_no
        node_with_gradients = DistCommon._deserialize_proto_node_gradients(request.node_gradients)
        return node_with_gradients, acc_no


class ParameterServiceClient:

    def __init__(self, ps_host):
        self.stub = parameter_server_pb2_grpc.ParameterServiceStub(grpc.insecure_channel(ps_host))
        print('[GRPC] Connected to parameter service: {}'.format(ps_host))

    def variable_weights_init(self, var_weights_dict):
        init_req = DistCommon._serialize_proto_variable_weights(var_weights_dict)
        init_resp = self.stub.VariableWeightsInit(init_req)
        duplicated_var_weights_dict = DistCommon._deserialize_proto_variable_weights(init_resp)
        return duplicated_var_weights_dict

    def push_gradients(self, acc_gradients, acc_no):
        proto_node_gradients = DistCommon._serialize_proto_node_gradients(acc_gradients)
        proto_node_gradients.acc_no = acc_no
        push_req = parameter_server_pb2.ParameterPushReq(node_gradients=proto_node_gradients)
        resp = self.stub.Push(push_req)
        return resp

    def pull_gradients(self, nodes_name=None):
        pull_req = parameter_server_pb2.ParameterPullReq()
        pull_resp = self.stub.Pull(pull_req)
        node_gradients = DistCommon._deserialize_proto_node_gradients(pull_resp.node_gradients)
        return node_gradients


class ParameterServiceServer:

    def __init__(self, cluster_conf, sync=True, max_workers=10):
        self.worker_num = len(cluster_conf['workers'])
        self.host = cluster_conf['ps'][0]
        self.sync = sync
        self.max_workers = max_workers

        self.server = grpc.server(ThreadPoolExecutor(max_workers=self.max_workers))
        parameter_server_pb2_grpc.add_ParameterServiceServicer_to_server(
            ParameterServer(self.worker_num, self.sync), self.server)
        self.server.add_insecure_port(self.host)

    def serve(self):
        self.server.start()
        print(f'[GRPC] Parameter server (mode: {self.sync}) running on {self.host} and worker num {self.worker_num}')
        try:
            while True:
                time.sleep(60 * 60 * 24)
        except KeyboardInterrupt:
            self.server.stop(0)
