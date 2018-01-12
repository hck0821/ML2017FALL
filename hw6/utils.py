import csv
import numpy as np
import pandas as pd
from itertools import accumulate

def read_data(filename):
    data = np.load(filename).astype('float')
    return data


def read_test_pair(filename):
    test_label = pd.read_csv(filename)
    test_pair = test_label[['image1_index','image2_index']].values
    return test_pair

# args: list of numpy ndarray (N, *, ...)
def shuffle(*args):
    idx = np.arange(args[0].shape[0])
    np.random.shuffle(idx)
    args = (arg[idx] for arg in args)
    return args


# args: list of numpy ndarray (*, D)
def normalize(*args):
    concat = np.concatenate(args)
    std = 255.
    #std = np.std(concat, axis=0)
    #std = np.tile(std, (concat.shape[0], 1))
    mean = np.mean(concat, axis=0)
    mean = np.tile(mean, (concat.shape[0], 1))
    concat = (concat - mean) / (std + 1e-20)
    if len(args) != 1:
        pos = accumulate(args, lambda acc, x: acc + len(x))
        args = tuple(np.split(concat, list(pos)))
    else:
        args = concat
    return args


def tfmt(s):
    m = s // 60
    s = s % 60

    h = m // 60
    m = m % 60

    if h != 0:
        f = '%d h %d m %d s' % (h, m, s)
    elif m != 0:
        f = '%d m %d s' % (m, s)
    else:
        f = '%d s' % s
    return f

if __name__ == '__main__':
    
    data = read_data('../data/image.npy')
    print(data.shape)
    '''
    data = normalize(data)
    print(data.shape)
    '''
    #test_pair = read_test_pair('../data/test_case.csv')
    #print(test_pair[0])
