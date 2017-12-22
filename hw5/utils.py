import csv
import numpy as np

rating_range = 5

def load_train_data(filename, normalize=True):
    data_file = open(filename, 'r')
    data = [row[1:4] for row in csv.reader(data_file, delimiter=',')]
    data = np.asarray(data[1:])
    id_pairs = data[:, 0:2].astype('int')
    ratings = data[:, 2].astype('float') / rating_range if normalize else data[:, 2].astype('float')
    return id_pairs, ratings


def load_test_data(filename):
    data_file = open(filename, 'r')
    data = [row[1:3] for row in csv.reader(data_file, delimiter=',')]
    data = np.asarray(data[1:], dtype='int')
    id_pairs = data[:, 0:2]
    return id_pairs

def load_class_data(filename):
    movie_file = open(filename, 'r', encoding='latin-1')
    header = movie_file.readline()
    movie2classes = []
    movie_ids = []
    class_dict = {}
    class_id = 0
    for infos in movie_file:
        movie_id = infos.strip().split(':')[0]
        classes = infos.strip().split(':')[-1].split('|')
        ids = []
        for c in classes:
            if c not in class_dict:
                class_dict[c] = class_id
                class_id += 1
            ids.append(class_dict[c])
        movie2classes.append(ids)
        movie_ids.append(movie_id)
    movie2ids = {}
    for i in range(len(movie2classes)):
        onehot = np.zeros((class_id))
        onehot[movie2classes[i]] = 1
        movie2ids[movie_ids[i]] = onehot
    #print(movie2ids)
    return movie2ids
#load_class_data('../data/movies.csv')

def shuffle(x, y):
    assert x.shape[0] == y.shape[0]
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    return x[indices], y[indices]


def normalize(x):
    mu = np.mean(x, axis=0)
    #mu = np.tile(mu, (x.shape[0]))
    sigma = np.std(x, axis=0)
    #sigma = np.tile(sigma, (x.shape[0]))
    x = (x - mu) / (sigma + 1e-20)
    return x, mu, sigma


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


