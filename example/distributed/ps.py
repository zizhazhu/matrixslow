import argparse

from sklearn.preprocessing import OneHotEncoder

import matrixslow as ms
from matrixslow.dataset.iris_data import gen_data


cluster_conf = {
    "ps": [
        "localhost:5000",
    ],
    "worker": [
        "localhost:5001",
        "localhost:5002",
        "localhost:5003",
    ]
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    return args


def train(index):
    features, labels = gen_data()

    one_hot_encoder = OneHotEncoder(sparse=False)
    one_hot_labels = one_hot_encoder.fit_transform(labels.reshape(-1, 1))

    x = ms.core.Variable(dim=(4, 1), init=False, trainable=False, name='x')
    y = ms.core.Variable(dim=(3, 1), init=False, trainable=False, name='y')

    model = ms.model.NN(input_size=4, layers=(10, 10), output_size=3)
    logits = model.forward(x)
    predict = ms.ops.Softmax(logits, name='predict')

    loss = ms.ops.loss.CrossEntropyWithSoftMax(logits, y)

    learning_rate = 0.01
    batch_size = 8
    n_epochs = 40
    optimizer = ms.optimizer.Adam(ms.default_graph, loss, learning_rate)

    trainer = ms.train.DistPSTrainer(optimizer, cluster_conf=cluster_conf,
                                     metric_ops=[ms.ops.metrics.Accuracy(predict, y),
                                                 ms.ops.metrics.Precision(predict, y),
                                                 ])
    trainer.train_and_test(train_dict={x: features, y: one_hot_labels}, test_dict={x: features, y: one_hot_labels},
                           n_epochs=n_epochs)


def main():
    args = get_args()
    if args.role == 'ps':
        server = ms.dist.ps.ParameterServiceServer(cluster_conf, sync=True)
        server.serve()
    else:
        worker_index = args.index
        train(worker_index)


if __name__ == '__main__':
    main()
