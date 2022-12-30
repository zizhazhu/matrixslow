import time
import threading
from concurrent.futures import ThreadPoolExecutor

import grpc

from ..dist import DistCommon
from ..proto import parameter_server_pb2, parameter_server_pb2_grpc


class ParameterService(parameter_server_pb2_grpc.ParameterServiceServicer):

    def __init__(self, worker_num, sync=True):
        self.node_gradients_cache = dict()
        self.variable_weights_cache = dict()

        self.sync = sync
        self.worker_num = worker_num
        self.cur_push_num = 0
        self.cur_pull_num = self.worker_num

        self.cond = threading.Condition()
        self.push_lock = threading.Lock()
        self.init_lock = threading.Lock()
        self.is_init = False

        self.acc_no = 0

    def Push(self, request, context):
        node_with_gradients, acc_no = self._deserialize_push_req(request)
        if self.sync:
            self._push_sync(node_with_gradients, acc_no)
        else:
            self._push_async(node_with_gradients, acc_no)

        return parameter_server_pb2.ParameterPushResp()

    def _push_sync(self, node_with_gradients, acc_no):
        if self.cond.acquire():
            while self.cur_pull_num != self.worker_num:
                self.cond.wait()

            self.cur_push_num += 1
            self._update_node_gradients_cache(node_with_gradients)
            self.acc_no += acc_no
            if self.cur_push_num >= self.worker_num:
                self.cur_pull_num = 0
                self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()

    def _push_async(self, node_with_gradients, acc_no):
        self.push_lock.acquire()
        self._update_node_gradients_cache(node_with_gradients)
        self.acc_no += acc_no
        self.push_lock.release()

    def _update_node_gradients_cache(self, node_with_gradients):
        for node, gradients in node_with_gradients.items():
            if node in self.node_gradients_cache:
                self.node_gradients_cache[node] += gradients
            else:
                self.node_gradients_cache[node] = gradients

    def Pull(self, request, context):
        if self.sync:
            resp = self._pull_sync()
        else:
            resp = self._pull_async()

        return resp

    def _pull_sync(self):
        resp = None
        if self.cond.acquire():
            while self.cur_pull_num != self.worker_num:
                self.cond.wait()

            self.cur_pull_num += 1
            self._gradients_cache_mean()
            resp = self._serialize_pull_resp()
            if self.cur_pull_num >= self.worker_num:
                self.cur_push_num = 0
                self._reset_gradients_cache()
                self.cond.notify_all()
            self.cond.release()
        else:
            self.cond.wait()
        return resp

    def _gradients_cache_mean(self):
        if self.acc_no != 0:
            for node, gradients in self.node_gradients_cache.items():
                self.node_gradients_cache[node] = gradients / self.acc_no
            self.acc_no = 0

    def _pull_async(self):
        self.push_lock.acquire()
        self._gradients_cache_mean()
        resp = self._serialize_pull_resp()
        self._reset_gradients_cache()
        self.push_lock.release()
        return resp

    def _deserialize_push_req(self, request):
        acc_no = request.gradients.acc_no
        node_with_gradients = DistCommon._deserialize_proto_node_gradients(request.gradients)
        return node_with_gradients, acc_no

    def _serialize_pull_resp(self):
        proto_node_gradients = DistCommon._serialize_proto_node_gradients(self.node_gradients_cache)
        resp = parameter_server_pb2.ParameterPullResp(gradients=proto_node_gradients)
        return resp

    def _reset_gradients_cache(self):
        self.node_gradients_cache.clear()

    def VariableWeightsInit(self, request, context):
        self.init_lock.acquire()
        # choose the first worker to init
        try:
            if not self.is_init:
                self.variable_weights_cache = DistCommon._deserialize_proto_variable_weights(request)
                print('[INIT] Parameter service variable weights initialized')
        except Exception as e:
            raise e
        else:
            resp = DistCommon._serialize_proto_variable_weights(self.variable_weights_cache)
            self.is_init = True
        finally:
            self.init_lock.release()

        return resp


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
        push_req = parameter_server_pb2.ParameterPushReq(gradients=proto_node_gradients)
        resp = self.stub.Push(push_req)
        return resp

    def pull_gradients(self, nodes_name=None):
        pull_req = parameter_server_pb2.ParameterPullReq()
        pull_resp = self.stub.Pull(pull_req)
        node_gradients = DistCommon._deserialize_proto_node_gradients(pull_resp.gradients)
        return node_gradients


class ParameterServiceServer:

    def __init__(self, cluster_conf, sync=True, max_workers=10):
        self.worker_num = len(cluster_conf['worker'])
        self.host = cluster_conf['ps'][0]
        self.sync = sync
        self.max_workers = max_workers

        self.server = grpc.server(ThreadPoolExecutor(max_workers=self.max_workers))
        parameter_server_pb2_grpc.add_ParameterServiceServicer_to_server(
            ParameterService(self.worker_num, self.sync), self.server)
        self.server.add_insecure_port(self.host)

    def serve(self):
        self.server.start()
        print(f'[GRPC] Parameter server (mode: {self.sync}) running on {self.host} and worker num {self.worker_num}')
        try:
            while True:
                time.sleep(60 * 60 * 24)
        except KeyboardInterrupt:
            self.server.stop(0)
