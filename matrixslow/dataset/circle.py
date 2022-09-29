import numpy as np
from sklearn.datasets import make_circles


def gen_data(n=200, noise_dimension=0):
    x, y = make_circles(n, noise=0.1, factor=0.2)
    y = y * 2 - 1
    if noise_dimension:
        noises = np.random.normal(0.0, 0.2, (n, noise_dimension))
        x = np.concatenate([x, noises], axis=1)
    return x, y
