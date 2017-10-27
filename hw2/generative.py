import os
import sys
import time
import numpy as np

import utils

class Generative_Model():
    def __init__(self, n_parms):
        self.n_parms = n_parms
        self.num = np.zeros(2)
        self.mu = np.zeros((2, n_parms))
        self.shared_sigma = np.zeros((n_parms, n_parms))

    def predict(self, x):
        inv_sigma = np.linalg.inv(self.shared_sigma)
        w = np.dot((self.mu[0] - self.mu[1]), inv_sigma)
        x = x.T
        b = (-0.5) * np.dot(np.dot([self.mu[0]], inv_sigma), self.mu[0]) + \
            (0.5) * np.dot(np.dot([self.mu[1]], inv_sigma), self.mu[1]) + \
            np.log(self.num[0]/self.num[1])
        a = np.dot(w, x) + b
        pred = utils.sigmoid(a)
        pred = np.around(pred)
        return pred

    def evaluate(self, y, pred):
        result = (y == pred).astype('float')
        acc = result.sum() / y.shape[0]
        return acc

    def fit(self, x, y):
        self.shared_sigma = np.zeros((self.n_parms, self.n_parms))
        for i in range(2): 
            cls = x[(y == 1 - i).flatten()]
            self.num[i] = cls.shape[0]
            self.mu[i] = np.mean(cls, axis=0)
            sigma = np.mean([np.dot(np.transpose([v - self.mu[i]]), [v - self.mu[i]]) for v in cls], axis=0)
            self.shared_sigma += (self.num[i] / x.shape[0]) * sigma
        pred = self.predict(x)
        acc = self.evaluate(y, pred)
        print('accuracy of training: %.3f' % acc)
        



if __name__ == '__main__':
    tr_x, tr_y, tt_x = utils.load_data(sys.argv[1], sys.argv[2], sys.argv[3])
    tr_x, tt_x = utils.normalize(tr_x, tt_x)
    
    valid_ratio = 0.0
    if valid_ratio != 0.0:
        valid_size = int((tr_x.shape[0]*valid_ratio) // 1)
        tr_x, tr_y = utils.shuffle(tr_x, tr_y)
        val_x, val_y = tr_x[valid_size:], tr_y[valid_size:]
        tr_x, tr_y = tr_x[0:valid_size], tr_y[0:valid_size]

    model = Generative_Model(tr_x.shape[1])
    print('Start training...')
    model.fit(tr_x, tr_y)
    if valid_ratio != 0.0:
        pred = model.predict(val_x)
        acc = model.evaluate(val_y, pred)
        print('accuracy of validation: %.3f' % acc)

    print('Predict testing data...')
    pred = model.predict(tt_x)
    outfile = open(sys.argv[4], 'w')
    outfile.write('id,label\n')
    for i, v in enumerate(pred):
        outfile.write('%d,%d\n' % (i + 1, v))
    outfile.close()
                                   
