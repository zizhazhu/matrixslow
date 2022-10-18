import numpy as np
from scipy import signal


def get_sequence_data(dimension=10, length=10):
    candidates = []
    candidates.append(np.sin(np.arange(0, 10, 10 / length)).reshape(-1, 1))
    candidates.append(np.array(signal.square(np.arange(0, 10, 10 / length))).reshape(-1, 1))

    data = []
    for i in range(2):
        candidate = candidates[i]
        # 基于原来的函数扩展出500种随机
        for j in range(500):
            # 原来的y加上dimension维度的随机值，扩展维度
            sequence = candidate + np.random.normal(0, 0.6, (len(candidate), dimension))
            label = np.array([int(i == k) for k in range(2)])
            data.append(np.c_[sequence.reshape(1, -1), label.reshape(1, -1)])
    data = np.concatenate(data, axis=0)
    np.random.shuffle(data)
    return data[:, :-2].reshape(-1, length, dimension), data[:, -2:]

