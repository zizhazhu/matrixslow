import grpc
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import matrixslow as ms
from matrixslow_serving.proto.serving_pb2_grpc import MatrixSlowServingServicer
import matrixslow_serving.proto.serving_pb2_grpc as serving_pb2_grpc
import matrixslow_serving.proto.serving_pb2 as serving_pb2


class MatrixSlowServingService(MatrixSlowServingServicer):

    def __init__(self, model_path):
        saver = ms.train.Saver(model_path)
        _, service = saver.load()
        self.input_node = ms.default_graph.get_node_by_name(service['input']['name'])
        self.output_node = ms.default_graph.get_node_by_name(service['output']['name'])

    def Predict(self, request, context):
        matrices = MatrixSlowServingService.deserialize(request)
        inference_response_matrices = self._inference(matrices)
        predict_response = MatrixSlowServingService.serialize(inference_response_matrices)
        return predict_response

    def _inference(self, matrices):
        inference_response_matrices = []
        for matrix in matrices:
            self.input_node.set_value(matrix.T)
            self.output_node.forward()
            inference_response_matrices.append(self.output_node.value)
        return inference_response_matrices

    @staticmethod
    def deserialize(request):
        matrices = []
        for proto_mat in request.matrix:
            dim = tuple(proto_mat.dim)
            mat = np.mat(proto_mat.value, dtype=float).reshape(dim)
            matrices.append(mat)
        return matrices

    @staticmethod
    def serialize(response):
        proto_response = serving_pb2.PredictResponse()
        for mat in response:
            proto_mat = proto_response.matrix.add()
            proto_mat.dim.extend(list(mat.shape))
            proto_mat.value.extend(np.array(mat).flatten())
        return proto_response


class MatrixSlowServer:

    def __init__(self, host, model_path, max_workers=10):
        self.host = host
        self.model_path = model_path
        self.max_workers = max_workers
        self.server = grpc.server(ThreadPoolExecutor(max_workers=self.max_workers))
        serving_pb2_grpc.add_MatrixSlowServingServicer_to_server(MatrixSlowServingService(model_path), self.server)
        self.server.add_insecure_port(self.host)

    def serve(self):
        self.server.start()
        print(f'Serving on {self.host}')
        try:
            while True:
                time.sleep(60 * 60 * 24)
        except KeyboardInterrupt:
            self.server.stop(0)
