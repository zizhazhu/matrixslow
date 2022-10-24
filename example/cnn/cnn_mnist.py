from sklearn.preprocessing import OneHotEncoder

import matrixslow as ms
from matrixslow.dataset.mnist import gen_data

features, labels = gen_data()
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_label = one_hot_encoder.fit_transform(labels.reshape(-1, 1))

img_shape = (28, 28)

x = ms.core.Variable(dim=(28, 28), init=False, trainable=False)
y = ms.core.Variable(dim=(10, 1), init=False, trainable=False)

conv1 = ms.layer.conv([x], img_shape, 3, (5, 5), "ReLU")
