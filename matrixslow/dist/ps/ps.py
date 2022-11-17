import threading

from ..dist import DistCommon
from ..proto import parameter_server_pb2, parameter_server_pb2_grpc

class ParameterServer(parameter_server_pb2_grpc.ParameterServiceServicer):

    def __init__(self, worker_num):
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

