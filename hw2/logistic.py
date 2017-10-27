from __future__ import print_function
import os
import sys
import time
import numpy as np

import utils


sel_range = [0, 1, 3, 4, 5]

def select_attributes(data, sel_range):
    sel_data = np.concatenate((data, np.log(data[:, sel_range] + 1e-10),
                               data[:, sel_range]**2, data[:, sel_range]**3, 
                               data[:, sel_range]**4, data[:, sel_range]**5,
                               data[:, 6:] * data[:, 5].reshape((-1, 1)),
                               (data[:, 0] * data[:, 3]).reshape((-1, 1)),
                               (data[:, 0] * data[:, 5]).reshape((-1, 1)),
                               (data[:, 0] * data[:, 5]).reshape((-1, 1)) ** 2,
                               (data[:, 3] * data[:, 5]).reshape((-1, 1)),
                               (data[:, 3] - data[:, 4]).reshape((-1, 1)),
                               (data[:, 3] - data[:, 4]).reshape((-1, 1)) ** 3
                               ), axis=1) # bias
    return sel_data


class Logistic_Regression():
    def __init__(self, n_parms):
        self.n_parms = n_parms
        self.w = np.zeros(n_parms)
        self.r = 0.0

    def loss(self, y, pred):
        return -1 * (np.dot(y, np.log(pred + 1e-20)) + \
                     np.dot((1 - y), np.log(1 - pred + 1e-20))) / y.shape[0] + \
               self.r * np.sum(self.w[:-1]**2)

    def predict(self, x, out=False):
        pred =  utils.sigmoid(np.dot(x, self.w))
        if out:
            pred = np.around(pred)
        return pred

    def evaluate(self, y, pred):
        pred = np.around(pred)
        result = (y == pred).astype('float')
        acc = result.sum() / y.shape[0]
        return acc

    def fit(self, x, y, epoch=2000, lr=0.05, r=0.0):
        self.r = r
        s_grad = np.zeros(self.n_parms)
        for i in range(epoch):
            pred = self.predict(x)
            loss = self.loss(y, pred)
            err = pred - y
            grad = np.dot(x.T, err)
            grad[:-1] += self.r * self.w[:-1]
            s_grad += grad**2
            ada = np.sqrt(s_grad)
            self.w = self.w - lr * grad/ada
            if  (i + 1) % 100 == 0:
                acc = self.evaluate(y, pred)
                print('epoch: %4d | loss: %.3f | acc: %.3f' % (i + 1, loss, acc))

if __name__ == '__main__':
    tr_x, tr_y, tt_x = utils.load_data(sys.argv[1], sys.argv[2], sys.argv[3])
    tr_x = select_attributes(tr_x, sel_range)
    tt_x = select_attributes(tt_x, sel_range)
    
    tr_x, tt_x = utils.normalize(tr_x, tt_x)
    tr_x = np.concatenate((tr_x, np.ones((tr_x.shape[0], 1))), axis=1)
    tt_x = np.concatenate((tt_x, np.ones((tt_x.shape[0], 1))), axis=1)
    
    valid_ratio = 0.0
    if valid_ratio != 0.0:
        valid_size = int((tr_x.shape[0]*valid_ratio) // 1)
        tr_x, tr_y = utils.shuffle(tr_x, tr_y)
        val_x, val_y = tr_x[valid_size:], tr_y[valid_size:]
        tr_x, tr_y = tr_x[0:valid_size], tr_y[0:valid_size]

    model = Logistic_Regression(tr_x.shape[1])
    
    tStart = time.time()
    print('Start training...\n')
    model.fit(tr_x, tr_y)
    tEnd = time.time()
    if valid_ratio != 0.0:
        pred = model.predict(val_x)
        acc = model.evaluate(val_y, pred)
        print('acc of validation: %3f' % acc)
    print('cost %.3f sec\n' % (tEnd - tStart))
    
    print('Predict testing data...\n')
    pred = model.predict(tt_x, out=True)
    outfile = open(sys.argv[4], 'w')
    outfile.write('id,label\n')
    for i, v in enumerate(pred):
        outfile.write('%d,%d\n' % (i + 1, v))
    outfile.close()
