import sys
import csv
import time
import numpy as np

attrs = {'AMB_TEMP':0, 'CH4':1, 'CO':2, 'NMHC':3, 'NO':4, 'NO2':5,
        'NOx':6, 'O3':7, 'PM10':8, 'PM2.5':9, 'RAINFALL':10, 'RH':11,
        'SO2':12, 'THC':13, 'WD_HR':14, 'WIND_DIREC':15, 'WIND_SPEED':16, 'WS_HR':17}
mono_attrs = ['CO', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'SO2', 
              'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
quad_attrs = ['PM10', 'PM2.5']
sel_attrs = [mono_attrs, quad_attrs]
#mono_attrs = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
#        'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH',
#        'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']
#sel_attrs = [mono_attrs]

n_attrs = len(attrs)
period = 9

def load_trData(filename):
    trfile = open(filename, 'r', encoding='big5')
    data = np.asarray([row[3:] for idx, row in enumerate(csv.reader(trfile, delimiter=',')) if idx != 0])
    data[data == 'NR'] = 0.0

    x = []
    y = []
    #split by each month
    for i in range(0, data.shape[0], n_attrs*20):
        days = np.vsplit(data[i:i+n_attrs*20], 20) # shape: 20 * (18, 24)
        month = np.concatenate(days, axis=1) # shape: (18, 24*20)
        
        del_idx = [idx for idx, dat in enumerate(month[attrs['PM2.5']]) if dat == '-1']
        month = np.delete(month, del_idx, axis=1)

        for t in range(0, month.shape[1]-period):
            x.append(month[:, t:t+period].flatten())
            y.append(month[attrs['PM2.5']][t+period])
    
    return np.asarray(x, dtype='float'), np.asarray(y, dtype='float')


def load_ttData(filename):
    ttfile = open(filename, 'r', encoding='big5')
    data = np.asarray([row[2:] for row in csv.reader(ttfile, delimiter=',')])
    data[data == 'NR'] = 0.0
    
    x = [data[i:i+n_attrs].flatten() for i in range(0, data.shape[0], n_attrs)]
    
    return np.asarray(x, dtype='float')


def select_attributes(data, sel_attrs):
    sel_range = []
    for i in range(len(sel_attrs)):
        sel_range.append([])
        for attr in sel_attrs[i]:
            sel_range[i] += list(range(attrs[attr]*period, (attrs[attr]+1)*period))
    #concatenate and add bias
    sel_data = np.concatenate((data[:, sel_range[0]], data[:, sel_range[1]]**2, 
                               np.ones((data.shape[0], 1))), axis=1)
    #sel_data = np.concatenate((data[:, sel_range[0]], np.ones((data.shape[0], 1))), axis=1)
    return sel_data


class Linear_Regression():
    def __init__(self, n_parms):
        self.n_parms = n_parms
        self.w = np.zeros(n_parms)
        self.r = 0

    def loss(self, err):
        return np.sqrt(np.mean(err**2) + self.r * np.sum(self.w[:-1]**2)) # without bias

    def predict(self, x):
        return np.dot(x, self.w)

    def fit(self, x, y, epoch=50000, lr=0.5, r=0.0):
        self.r = r
        s_grad = np.zeros(self.n_parms)
        for i in range(epoch):
            predict = self.predict(x)
            err = predict - y
            loss = self.loss(err)
            grad = np.dot(x.T, err)
            grad[:-1] += self.r * self.w[:-1]
            s_grad += grad**2
            ada = np.sqrt(s_grad)
            self.w = self.w - lr * grad/ada

            print('epoch: %05d | loss: %f' % (i, loss))



if __name__ == "__main__":
    tr_x, tr_y = load_trData(sys.argv[1])
    tt_x = load_ttData(sys.argv[2])
    tr_x = select_attributes(tr_x, sel_attrs)
    tt_x = select_attributes(tt_x, sel_attrs)

    tStart = time.time()
    model = Linear_Regression(tr_x.shape[1])
    model.fit(tr_x, tr_y, epoch=100000)
    tEnd = time.time()
    print('cost %f sec' % (tEnd - tStart))

    predict = model.predict(tt_x)
    with open(sys.argv[3], 'w') as outfile:
        outfile.write('id,value\n')
        for idx, p in enumerate(predict):
            outfile.write('id_%d,%f\n' % (idx, p))
