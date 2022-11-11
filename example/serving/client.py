import grpc
import numpy as np

from matrixslow_serving import serving_pb2, serving_pb2_grpc


class MatrixSlowServingClient:
    def __init__(self, host):
        self.stub = serving_pb2_grpc.MatrixSlowServingStub(grpc.insecure_channel(host))
        print(f'Connected to {host}')

    def predict(self, matrices):
        request = serving_pb2.PredictRequest()
        for mat in matrices:
            proto_mat = request.matrix.add()
            proto_mat.dim.extend(list(mat.shape))
            proto_mat.value.extend(np.array(mat).flatten())
        response = self.stub.Predict(request)
        return response


if __name__ == '__main__':
    from matrixslow.dataset.iris_data import gen_data
    features, labels = gen_data()
    host = 'localhost:50051'
    client = MatrixSlowServingClient(host)
    for index in range(len(features)):
        feature = features[index]
        label = labels[index]
        response = client.predict([feature])
        print(f'Feature: {feature}, Label: {label}, Predict: {response.matrix[0].value}')

