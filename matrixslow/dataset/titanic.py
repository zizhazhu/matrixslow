import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def gen_data(file='./data/titanic.csv'):
    data = pd.read_csv(file).drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False)

    pclass = onehot_encoder.fit_transform(label_encoder.fit_transform(data["Pclass"].fillna(0)).reshape(-1, 1))
    sex = onehot_encoder.fit_transform(label_encoder.fit_transform(data["Sex"].fillna(0)).reshape(-1, 1))
    embarked = onehot_encoder.fit_transform(label_encoder.fit_transform(data["Embarked"].fillna(0)).reshape(-1, 1))

    features = np.concatenate([pclass,
                               sex,
                               data[["Age"]].fillna(0),
                               data[["SibSp"]].fillna(0),
                               data[["Parch"]].fillna(0),
                               data[["Fare"]].fillna(0),
                               embarked,
                               ], axis=1)
    labels = data["Survived"].values * 2 - 1
    return features, labels
