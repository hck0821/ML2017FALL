from __future__ import print_function
import sys
import time
import argparse
import numpy as np
import pickle as pk
from gensim.models import Word2Vec
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from model import text_sentiment_classifier, bow_dnn
from utils import load_train_data, load_nolabel_data, load_test_data, shuffle, tfmt


parser = argparse.ArgumentParser(prog='main.py', description='ML2017-hw4 Text Sentiment Classifier')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--semi', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--batch', type=int, default=512)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--mode', type=str, default='word2vec')
parser.add_argument('--model', type=str, default='GRU')
parser.add_argument('--token_file', type=str, default='./token.pk')
parser.add_argument('--test_file', type=str, default='../data/testing_data.txt')
parser.add_argument('--train_file', type=str, default='../data/training_label.txt')
parser.add_argument('--nolabel_file', type=str, default='../data/training_nolabel.txt')
parser.add_argument('--vector_dict', type=str, default='./200dim_dict.txt')
parser.add_argument('--prefix', type=str, default='unnamed')
parser.add_argument('--save_dir', type=str, default='./')
parser.add_argument('--result_file', type=str, default='./hw4.csv')
parser.add_argument('--vector_dim', type=int, default=200)
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--maxlen', type=int, default=None)
args = parser.parse_args()


if not args.test and not args.train and not args.semi:
    print ('Error: can\'t find argument --train, --test or --semi.')
    sys.exit()

if args.mode == 'word2vec':
    if args.train:
        print('Load Data...', end=' ')
        sys.stdout.flush()
        tStart = time.time()
        test_corpus = load_test_data(args.test_file, mode='train')
        train_corpus, _ = load_train_data(args.train_file, mode='train')
        nolabel_corpus = load_nolabel_data(args.nolabel_file, mode='train')
        all_corpus = test_corpus + train_corpus + nolabel_corpus
        tEnd = time.time()
        print('cost %s' % tfmt(tEnd-tStart))
       
        print('Train Word2vec Model...', end=' ')
        sys.stdout.flush()
        tStart = time.time()
        word2vec = Word2Vec(all_corpus, size=args.vector_dim, window=5, min_count=3)
        tEnd = time.time()
        print('cost %s' % tfmt(tEnd-tStart))
        
        print('Save Word2Vec as Dictionary...')
        vector_dict = open(args.vector_dict, 'w')
        for idx, word in enumerate(word2vec.wv.index2word):
            print(word, end=',', file=vector_dict)
            print(*word2vec[word], file=vector_dict)
        vector_dict.close()
        
        print('Save Token...')
        all_corpus = [' '.join(word_seq) for word_seq in all_corpus]
        tokenizer = Tokenizer(filters='\t\n')
        tokenizer.fit_on_texts(all_corpus)
        pk.dump(tokenizer, open(args.token_file, 'wb'))
    
    if args.test:
        print('Load Word2Vec Dictionary...')
        vector_dict = open(args.vector_dict, 'r')
        word2vec = {}
        for line in vector_dict:
            line = line.strip().split(',')
            word = line[0]
            vector_seq = np.asarray(line[1].split(), dtype='float')
            word2vec[word] = vector_seq
            
        while True:
            word = input('word2vec$ ')
            if word.lower() == 'exit':
                sys.exit()
            elif word in word2vec:
                print('Word Embedding Vector:')
                print(word2vec[word], end='\n\n')
            else:
                print('\'', word, '\'', ' Is Not in Corpus!', sep='', end='\n\n')


if args.mode == 'classify':
    params = args.save_dir + args.prefix + '.h5'
    logger = args.save_dir + args.prefix + '_log.csv'
    
    print('Load Word2Vec Dictionary and Token...')
    vector_dict = open(args.vector_dict, 'r')
    word2vec = {}
    for line in vector_dict:
        line = line.strip().rsplit(',', 1)
        word = line[0]
        vector_seq = np.asarray(line[1].split(), dtype='float')
        word2vec[word] = vector_seq
    tokenizer = pk.load(open(args.token_file, 'rb'))

    print('Construct Embedding Matrix...')
    if args.model == 'bow':
        num_word = 20000
        embedding_matrix = np.eye(N=num_word+1, M=num_word, k=-1)
    else:
        embedding_matrix = np.zeros((len(tokenizer.word_index)+1, args.vector_dim), dtype='float')
        for word, idx in tokenizer.word_index.items():
            if word in word2vec:
                embedding_matrix[idx] = word2vec[word]
    
    if args.model == 'bow':
        print('Select %s Model' % args.model)
        model = bow_dnn(embedding_matrix, verbose=1)
    else:
        print('Select Text Setiment Classifier with %s Cell...' % args.model.upper())
        model = text_sentiment_classifier(embedding_matrix, cell=args.model, verbose=1)

    if args.train or args.semi:
        print('\nLoad Training Data...', end=' ')
        sys.stdout.flush()
        tStart = time.time()
        train_corpus, train_y = load_train_data(args.train_file, mode='train')
        tEnd = time.time()
        print('cost %s' % tfmt(tEnd-tStart))
        
        maxlen = args.maxlen if args.maxlen != None else max(len(word_seq) for word_seq in train_corpus)
        print('Convert Sentences to Vector Sequence...')
        train_corpus = [' '.join(word_seq) for word_seq in train_corpus]
        train_x = tokenizer.texts_to_sequences(train_corpus)
        train_x = pad_sequences(train_x, maxlen, padding='post', truncating='post')
        train_y = np.asarray(train_y, dtype='int')
        del train_corpus

        if args.valid_ratio != 0:
            valid_size = int(len(train_x) * args.valid_ratio)
            print('Split %d/%d validation data...' % (valid_size, len(train_x)))
            train_x, train_y = shuffle(train_x, train_y)
            valid_x, valid_y = train_x[:valid_size], train_y[:valid_size]
            train_x, train_y = train_x[valid_size:], train_y[valid_size:]
        
        csvlogger = CSVLogger(logger)
        earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
        checkpoint = ModelCheckpoint(params, monitor='val_acc', save_best_only=True,
                                     save_weights_only=True, verbose=0, mode='max')
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Start Training...')
        model.fit(train_x, train_y, batch_size=args.batch, epochs=args.epoch, verbose=1, shuffle=True,
                  validation_data=(valid_x, valid_y), callbacks=[checkpoint, earlystopping, csvlogger])
    
        if args.semi:
            print('\nLoad Unlabeled Data...', end=' ')
            sys.stdout.flush()
            tStart = time.time()
            nolabel_corpus = load_nolabel_data(args.nolabel_file, mode='train')
            tEnd = time.time()
            print('cost %s' % tfmt(tEnd-tStart))
            
            print('Convert Sentences to Vector Sequence...')
            nolabel_corpus = [' '.join(word_seq) for word_seq in nolabel_corpus if len(word_seq) <= maxlen]
            nolabel_x = tokenizer.texts_to_sequences(nolabel_corpus)
            nolabel_x = pad_sequences(nolabel_x, maxlen, padding='post', truncating='post')
            del nolabel_corpus

            print('Predict Unlabeled Data...', end=' ')
            sys.stdout.flush()
            tStart = time.time()
            pred = model.predict(nolabel_x, batch_size=args.batch).reshape(-1)
            indices = np.logical_or(pred <= 0.05, pred >= 0.95)
            nolabel_x = nolabel_x[indices, :]
            nolabel_y = np.around(pred[indices]).astype('int')
            del pred
            tEnd = time.time()
            print('cost %s' % tfmt(tEnd-tStart))
            
            print('Augment Training Data...')
            train_x = np.concatenate((train_x, nolabel_x))
            train_y = np.concatenate((train_y, nolabel_y))
            del nolabel_x, nolabel_y
            
            csvlogger = CSVLogger(logger, append=True)
            earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
            checkpoint = ModelCheckpoint(params, monitor='val_acc', save_best_only=True,
                                         save_weights_only=True, verbose=0, mode='max')
            print('Start Semi-Supervised Learning...')
            model = text_sentiment_classifier(embedding_matrix, cell=args.model, verbose=0)
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(train_x, train_y, batch_size=args.batch, epochs=args.epoch, verbose=1, shuffle=True,
                      validation_data=(valid_x, valid_y), callbacks=[checkpoint, earlystopping, csvlogger])

    if args.test:
        print('\nLoad Text Setiment Classifier Parameters...')
        model.load_weights(params)
        
        print('Load Testing Data...', end=' ')
        sys.stdout.flush()
        tStart = time.time()
        test_corpus = load_test_data(args.test_file, mode='test')
        tEnd = time.time()
        print('cost %s' % tfmt(tEnd-tStart))
        
        print('Convert Sentences to Vector Sequence...')
        test_corpus = [' '.join(word_seq) for word_seq in test_corpus if len(word_seq) <= args.maxlen]
        test_x = tokenizer.texts_to_sequences(test_corpus)
        test_x = pad_sequences(test_x, args.maxlen, padding='post', truncating='post')
        del test_corpus
       
        print('Start Testing...', end=' ')
        sys.stdout.flush()
        tStart = time.time()
        pred = model.predict(test_x, batch_size=args.batch).reshape(-1)
        pred = np.around(pred).astype('int')
        tEnd = time.time()
        print('cost %s' % tfmt(tEnd-tStart))
        
        print('Save Prediction...')
        outfile = open(args.result_file, 'w')
        print('id,label', file=outfile)
        for i, p in enumerate(pred):
            print(i, p, sep=',', file=outfile)
        
