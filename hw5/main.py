from __future__ import print_function
import sys
import time
import argparse
import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model

from model import Matrix_Factorization, DNN
from utils import load_train_data, load_test_data,load_class_data, normalize, shuffle, tfmt, rating_range


parser = argparse.ArgumentParser(prog='main.py', description='ML2017-hw5 Matrix_Factorization')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--batch', type=int, default=512)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--model', type=str, default='MF')
parser.add_argument('--test_file', type=str, default='../data/test.csv')
parser.add_argument('--train_file', type=str, default='../data/train.csv')
parser.add_argument('--prefix', type=str, default='unnamed')
parser.add_argument('--save_dir', type=str, default='./')
parser.add_argument('--result_dir', type=str, default='../result/')
parser.add_argument('--result_file', type=str, default=None)
parser.add_argument('--vector_dim', type=int, default=256)
parser.add_argument('--valid_ratio', type=float, default=0.1)
args = parser.parse_args()


if not args.test and not args.train:
    print ('Error: can\'t find argument --train or --test.')
    sys.exit()

params = args.save_dir + args.prefix + '.h5'
logger = args.save_dir + args.prefix + '_log.csv'

if args.train:
    print('Load Training Data...', end=' ')
    sys.stdout.flush()
    tStart = time.time()
    train_x, train_y = load_train_data(args.train_file, normalize=False)
    tEnd = time.time()
    print('cost %s' % tfmt(tEnd-tStart))
    
    if args.valid_ratio != 0:
        valid_size = int(len(train_x) * args.valid_ratio)
        print('Split %d/%d validation data...' % (valid_size, len(train_x)))
        train_x, train_y = shuffle(train_x, train_y)
        valid_x, valid_y = train_x[-valid_size:], train_y[-valid_size:]
        train_x, train_y = train_x[:-valid_size], train_y[:-valid_size]
    num_users = max(train_x[:, 0])
    num_movies = max(train_x[:, 1])
    print('Select %s Model' % args.model)
    if args.model == 'MF':
        model = Matrix_Factorization(num_users, num_movies, args.vector_dim, verbose=1)
    if args.model == 'DNN':
        model = DNN(num_users, num_movies, args.vector_dim, verbose=1)
    
    adam = Adam(lr=1e-4)
    csvlogger = CSVLogger(logger)
    earlystopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(params, monitor='val_loss', save_best_only=True,
                                 save_weights_only=True, verbose=0, mode='min')
    model.compile(loss='mse', optimizer=adam)
    print('Start Training...')
    model.fit([train_x[:,0], train_x[:,1]], train_y, batch_size=args.batch, epochs=args.epoch, verbose=1,
              shuffle=True, validation_data=([valid_x[:,0], valid_x[:,1]], valid_y),
              callbacks=[checkpoint, earlystopping, csvlogger])


if args.test:
    print('Load %s Model...' % args.model)
    if args.model == 'MF':
        model = Matrix_Factorization(6040, 3952, args.vector_dim, verbose=1)
    if args.model == 'DNN':
        model = DNN(num_users, num_movies, args.vector_dim, verbose=1)
    model.load_weights(params)
    
    print('Load Testing Data...', end=' ')
    sys.stdout.flush()
    tStart = time.time()
    test_x = load_test_data(args.test_file)
    tEnd = time.time()
    print('cost %s' % tfmt(tEnd-tStart))
    
   
    print('Start Testing...', end=' ')
    sys.stdout.flush()
    tStart = time.time()
    pred = model.predict([test_x[:,0], test_x[:,1]], batch_size=args.batch)
    pred = np.clip(pred, 1.0, 5.0)
    tEnd = time.time()
    print('cost %s' % tfmt(tEnd-tStart))
    
    print('Save Prediction...')
    if args.result_file == None:
        outfile = open(args.result_dir + args.prefix + '.csv', 'w')
    else:
        outfile = open(args.result_file, 'w')
    print('TestDataId,Rating', file=outfile)
    for i, p in enumerate(pred):
        print(i+1, *p, sep=',', file=outfile)
        
