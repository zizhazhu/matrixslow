import numpy as np

def gen_data(shuffle=True):
    male_heights = np.random.normal(171, 6, 500)
    female_heights = np.random.normal(158, 5, 500)

    male_weights = np.random.normal(70, 10, 500)
    female_weights = np.random.normal(57, 8, 500)

    male_bfrs = np.random.normal(16, 2, 500)
    female_bfrs = np.random.normal(22, 2, 500)

    male_labels = [1] * 500
    female_labels = [-1] * 500

    # 把所有男女性别的数据拼合在一起，并打乱顺序
    train_data = np.array([np.concatenate((male_heights, female_heights)),
                           np.concatenate((male_weights, female_weights)),
                           np.concatenate((male_bfrs, female_bfrs)),
                           np.concatenate((male_labels, female_labels)),
                           ]).T
    if shuffle:
        np.random.shuffle(train_data)
    return train_data
