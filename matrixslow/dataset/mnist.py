from sklearn.datasets import fetch_openml


def gen_data(folder='./data', num=5000):
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, data_home=folder)
    X = X[:num].to_numpy()
    y = y.astype(int)[:num].to_numpy()
    return X, y
