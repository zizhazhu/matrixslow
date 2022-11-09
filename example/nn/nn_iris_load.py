from sklearn.preprocessing import OneHotEncoder
import numpy as np

import matrixslow as ms
from matrixslow.dataset.iris_data import gen_data

features, labels = gen_data()

one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_labels = one_hot_encoder.fit_transform(labels.reshape(-1, 1))

saver = ms.train.Saver('./save/iris')
saver.load()

x = ms.default_graph.get_node_by_name('x')
predict = ms.default_graph.get_node_by_name('predict')

for index in range(len(features)):
    x.set_value(np.mat(features[index]).T)
    predict.forward()
    true_label = labels[index]
    print(f'Predict: {predict.value}, True: {true_label}')