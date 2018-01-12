from __future__ import print_function
import sys
import time
import argparse
import numpy as np
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model, Model
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import KMeans, MiniBatchKMeans

from model import seq2seq
from utils import read_data, read_test_pair, normalize, shuffle, tfmt


parser = argparse.ArgumentParser(prog='main.py', description='ML2017-hw6 Unsupervised Learning & Dimension Reduction')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--model', type=str, default='seq2seq')
parser.add_argument('--batch', type=int, default=512)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--latent', type=int, default=32)
parser.add_argument('--test_file', type=str, default='../data/test_case.csv')
parser.add_argument('--train_file', type=str, default='../data/image.npy')
parser.add_argument('--prefix', type=str, default='unnamed')
parser.add_argument('--save_dir', type=str, default='./')
parser.add_argument('--result_dir', type=str, default='../result/')
parser.add_argument('--result_file', type=str, default=None)
#parser.add_argument('--valid_ratio', type=float, default=0.1)
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
    train_x = read_data(args.train_file)
    #train_x = train_x / 255.
    train_x = normalize(train_x)
    tEnd = time.time()
    print('cost %s' % tfmt(tEnd-tStart))
    
    '''
    if args.valid_ratio != 0:
        valid_size = int(len(train_x) * args.valid_ratio)
        print('Split %d/%d validation data...' % (valid_size, len(train_x)))
        train_x = shuffle(train_x)
        valid_x = train_x[-valid_size:]
        train_x = train_x[:-valid_size]
    '''

    print('Select %s Model' % args.model)
    if args.model == 'seq2seq':
        model = seq2seq(train_x.shape[1], args.latent, verbose=1)
    
    adam = Adam(lr=1e-3)
    csvlogger = CSVLogger(logger)
    earlystopping = EarlyStopping(monitor='loss', patience=20, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(params, monitor='loss', save_best_only=True, verbose=0, mode='min')
    model.compile(loss='mse', optimizer=adam)
    print('Start Training...')
    model.fit(train_x, train_x, batch_size=args.batch, epochs=args.epoch, verbose=1,
              shuffle=True, callbacks=[checkpoint, earlystopping, csvlogger])
              #validation_data=([valid_x[:,0], valid_x[:,1]], valid_y))
              

if args.test:
    print('Load Testing Data...', end=' ')
    sys.stdout.flush()
    tStart = time.time()
    train_x = read_data(args.train_file)
    #train_x = train_x / 255.
    train_x = normalize(train_x)
    test_pair = read_test_pair(args.test_file)
    tEnd = time.time()
    print('cost %s' % tfmt(tEnd-tStart))
    
    print('Load %s Model...' % args.model)
    seq2seq = load_model(params)
    model = Model(inputs=seq2seq.layers[0].input, outputs=seq2seq.layers[-2].get_output_at(-1))
    model.summary()

    print('Generate Feature...', end=' ')
    sys.stdout.flush()
    tStart = time.time()
    feat = model.predict(train_x, batch_size=args.batch)

    #pca = PCA(n_components=128).fit(train_x)
    #feat = pca.transform(train_x)
    tEnd = time.time()
    print('cost %s' % tfmt(tEnd-tStart))
    
    print('Train KMeans Model...', end=' ')
    sys.stdout.flush()
    tStart = time.time()
    kmeans = KMeans(init='k-means++', n_clusters=2, random_state=32).fit(feat)
    tEnd = time.time()
    print('cost %s' % tfmt(tEnd-tStart))

    print('Predict Test Data...')
    test_x = feat[test_pair.reshape(-1)]
    pred = kmeans.predict(test_x).reshape((-1, 2))
    pred = np.asarray([p[0] == p[1] for p in pred], dtype='int')
    
    print('Save Prediction...')
    if args.result_file == None:
        outfile = open(args.result_dir + args.prefix + '.csv', 'w')
    else:
        outfile = open(args.result_file, 'w')
    
    print('ID,Ans', file=outfile)
    for i, p in enumerate(pred):
        print(i, p, sep=',', file=outfile)
