import pandas as pd
import numpy as np
import re

def load_data(filename):
    data_file = open(filename, 'r')
    header = data_file.readline()
    data = np.asarray([re.split(r' |,', line.strip()) for i, line in enumerate(data_file)])
    label = data[:, 0].astype('int')
    feat = data[:, 1:].astype('float') / 255.
    #feat = np.asarray([preprocess(f) for f in feat])
    return feat, label


def shuffle(x, y):
    assert x.shape[0] == y.shape[0]
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]


def normalize(tr_x, tt_x):
    concat_x = np.concatenate((tr_x, tt_x))
    mu = np.mean(concat_x, axis=0)
    mu = np.tile(mu, (concat_x.shape[0], 1))
    sigma = np.std(concat_x, axis=0)
    sigma = np.tile(sigma, (concat_x.shape[0], 1))
    concat_x = (concat_x - mu) / (sigma + 1e-20)

    tr_x = concat_x[0:tr_x.shape[0]]
    tt_x = concat_x[tr_x.shape[0]:]
    return tr_x, tt_x

def preprocess(img):
    img = img / 255.
    '''
    # ver2
    img = img / 255.
    img = img - 0.5
    img = img * 2.0
    # ver1
    mean = (0.0 + 255.) / 2
    rng = 255. - mean
    img = img - np.mean(img)
    abs_max = np.max(img) if np.max(img) > -np.min(img) else -np.min(img)
    if abs_max == 0:
        img = (img + mean) / 255.
    else:
        ratio = rng / abs_max
        img = (img*ratio + mean) / 255.
    '''
    return img

if __name__ == '__main__':
    train_x, train_y = load_data('../data/train.csv')
    print(train_x.shape, train_y.shape)

    test_x, _ = load_data('../data/test.csv')
    print(test_x.shape)

