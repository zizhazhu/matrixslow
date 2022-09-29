from sklearn.datasets import make_classification


def gen_data(n=200, dimension=20, noise_dimension=40):
    x, y = make_classification(n, dimension + noise_dimension, n_informative=dimension)
    y = y * 2 - 1
    return x, y
