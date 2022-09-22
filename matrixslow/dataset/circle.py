from sklearn.datasets import make_circles


def gen_data(n=200):
    x, y = make_circles(n, noise=0.1, factor=0.2)
    y = y * 2 - 1
    return x, y
