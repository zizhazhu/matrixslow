import pandas as pd
from sklearn.preprocessing import LabelEncoder


def gen_data(file='./data/Iris.csv'):
    data = pd.read_csv(file).drop("Id", axis=1)
    data = data.sample(len(data), replace=False)
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform(data["Species"])
    features = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

    return features, label
