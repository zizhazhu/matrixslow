import matrixslow as ms
from matrixslow.dataset.trigonometric import get_sequence_data


train_features, train_labels = get_sequence_data(10, 10)
print(train_features)

