import pandas as pd
import numpy as np

def load_data(trfile, labfile, ttfile):
    tr_x = pd.read_csv(trfile)
    tr_x = np.asarray(tr_x, dtype='float')
    tr_y = pd.read_csv(labfile)
    tr_y = np.asarray(tr_y, dtype='int')
    tr_y = np.squeeze(tr_y)
    tt_x = pd.read_csv(ttfile)
    tt_x = np.asarray(tt_x, dtype='float')
    return tr_x, tr_y, tt_x


def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-14, 1-1e-14)



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
