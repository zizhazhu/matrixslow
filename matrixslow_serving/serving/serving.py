import grpc
import numpy as np

from matrixslow_serving.proto.serving_pb2_grpc import MatrixSlowServingServicer

class MatrixSlowServingService(MatrixSlowServingServicer):

    def __init__(self):
        pass

    def predict(self, request, context):
        matrices = MatrixSlowServingService.deserialize(request)

    def _inference(self):
        pass

    @staticmethod
    def deserialize(request):
        matrices = []
        for proto_mat in request.data:
            dim = tuple(proto_mat.dim)
            mat = np.mat(proto_mat.value, dtype=float).reshape(dim)
            matrices.append(mat)
        return matrices

