import numpy as np
import time
import re


def sentence_to_words(sentence, mode='train'):
    sentence = ' ' + sentence + ' '
    if mode == 'train':
        sentence = re.sub(r'[^A-Za-z0-9,.!\'\?]', ' ', sentence)
    # digit 
    # preposition
    # abbreviation
    sentence = re.sub(r' isnt ', ' is not ', sentence)
    sentence = re.sub(r' arent ', ' are not ', sentence)
    sentence = re.sub(r' wasnt ', ' was not ', sentence)
    sentence = re.sub(r' werent ', ' were not ', sentence)
    sentence = re.sub(r' isn \' t ', ' is not ', sentence)
    sentence = re.sub(r' aren \' t ', ' are not ', sentence)
    sentence = re.sub(r' wasn \' t ', ' was not ', sentence)
    sentence = re.sub(r' weren \' t ', ' were not ', sentence)
    
    sentence = re.sub(r' dont ', ' do not ', sentence)
    sentence = re.sub(r' didnt ', ' did not ', sentence)
    sentence = re.sub(r' doesnt ', ' does not ', sentence)
    sentence = re.sub(r' don \' t ', ' do not ', sentence)
    sentence = re.sub(r' didn \' t ', ' did not ', sentence)
    sentence = re.sub(r' doesn \' t ', ' does not ', sentence)
    
    sentence = re.sub(r' cant ', ' can not ', sentence)
    sentence = re.sub(r' wont ', ' will not ', sentence)
    sentence = re.sub(r' hasnt ', ' has not ', sentence)
    sentence = re.sub(r' cannot ', ' can not ', sentence)
    sentence = re.sub(r' havent ', ' have not ', sentence)
    sentence = re.sub(r' cudnt ', ' could not ', sentence)
    sentence = re.sub(r' wudnt ', ' would not ', sentence)
    sentence = re.sub(r' shudnt ', ' should not ', sentence)
    sentence = re.sub(r' couldnt ', ' could not ', sentence)
    sentence = re.sub(r' wouldnt ', ' would not ', sentence)
    sentence = re.sub(r' shouldnt ', ' should not ', sentence)
    sentence = re.sub(r' can \' t ', ' can not ', sentence)
    sentence = re.sub(r' won \' t ', ' will not ', sentence)
    sentence = re.sub(r' hasn \' t ', ' has not ', sentence)
    sentence = re.sub(r' haven \' t ', ' have not ', sentence)
    sentence = re.sub(r' cudn \' t ', ' could not ', sentence)
    sentence = re.sub(r' wudn \' t ', ' would not ', sentence)
    sentence = re.sub(r' shudn \' t ', ' should not ', sentence)
    sentence = re.sub(r' couldn \' t ', ' could not ', sentence)
    sentence = re.sub(r' wouldn \' t ', ' would not ', sentence)
    sentence = re.sub(r' shouldn \' t ', ' should not ', sentence)
    

    sentence = re.sub(r' it \' s ', ' it is ', sentence)
    sentence = re.sub(r' he \' s ', ' he is ', sentence)
    sentence = re.sub(r' she \' s ', ' she is ', sentence)
    sentence = re.sub(r' how \' s ', ' how is ', sentence)
    sentence = re.sub(r' that \' s ', ' that is ', sentence)
    sentence = re.sub(r' what \' s ', ' what is ', sentence)
    sentence = re.sub(r' when \' s ', ' when is ', sentence)
    sentence = re.sub(r' where \' s ', ' where is ', sentence)
    sentence = re.sub(r' there \' s ', ' there is ', sentence)
    sentence = re.sub(r' nobody \' s ', ' nobody is ', sentence)
    sentence = re.sub(r' someone \' s ', ' someone is ', sentence)
    sentence = re.sub(r' everyone \' s ', ' everyone is ', sentence)
    sentence = re.sub(r' everything \' s ', ' everythind is ', sentence)
    sentence = re.sub(r' let \' s ', ' let us ', sentence)

    sentence = re.sub(r' m ', ' am ', sentence)
    sentence = re.sub(r' u ', ' you ', sentence)
    sentence = re.sub(r' r ', ' are ', sentence)
    sentence = re.sub(r' d ', ' would ', sentence)
    sentence = re.sub(r' re ', ' are ', sentence)
    sentence = re.sub(r' ll ', ' will ', sentence)
    sentence = re.sub(r' ve ', ' have ', sentence)
    sentence = re.sub(r' ive ', ' i have ', sentence)
    
    
    # punctuation
    #sentence = re.sub(r',', '', sentence)
    #sentence = re.sub(r'!', '', sentence)
    #sentence = re.sub(r'\.', '', sentence)
    sentence = re.sub(r'\'', '', sentence)
    #sentence = re.sub(r'\?', '', sentence)
    
    # possessive
    sentence = re.sub(r' s ', ' ', sentence)
    
    words = sentence.split()
    return words


def load_train_data(filename, mode):
    train_file = open(filename, 'r')
    
    labels = []
    corpus = []
    for line in train_file:
        line = line.strip().split(' ', 2)
        label = line[0]
        sentence = line[2]
        word_seq = sentence_to_words(sentence, mode=mode)
        if len(word_seq) != 0:
            labels.append(label)
            corpus.append(word_seq)
    return corpus, labels

def load_nolabel_data(filename, mode):
    nolabel_file = open(filename, 'r')
    
    corpus = []
    for line in nolabel_file:
        sentence = line.strip()
        word_seq = sentence_to_words(sentence, mode=mode)
        if len(word_seq) != 0:
            corpus.append(word_seq)
    return corpus


def load_test_data(filename, mode):
    test_file = open(filename, 'r')
    header = test_file.readline()

    corpus = []
    for line in test_file:
        sentence = line.strip().split(',', 1)[1]
        word_seq = sentence_to_words(sentence, mode=mode)
        if len(word_seq) != 0:
            corpus.append(word_seq)
    return corpus


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

    tr_x = concat_x[:tr_x.shape[0]]
    tt_x = concat_x[tr_x.shape[0]:]
    return tr_x, tt_x


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
    #tStart = time.time() 
    #corpus, labels = load_train_data('../data/training_label.txt', mode='train')
    #print(len(corpus), len(labels))
    #corpus, labels = load_train_data('../data/training_label.txt', mode='test')
    #print(len(corpus), len(labels))
    #corpus  = load_nolabel_data('../data/training_nolabel.txt', mode='train')
    #print(len(corpus))
    #corpus  = load_nolabel_data('../data/training_nolabel.txt', mode='test')
    #print(len(corpus))
    #corpus  = load_test_data('../data/testing_data.txt', mode='train')
    #print(len(corpus))
    #corpus  = load_test_data('../data/testing_data.txt', mode='test')
    #print(len(corpus))
    #print(max(len(s) for s in corpus))
    #tEnd = time.time()
    #print(tfmt(tEnd - tStart))
    import sys
    from gensim.models import Word2Vec
    print('Load Word2Vec Parameters...')
    word2vec = Word2Vec.load('../save/200dim.txt')
    
    def padded_vector_sequence(word_seq, maxlen):
        vector_seq = []
        zeros = [0.0]*200
        for word in word_seq:
            if word in word2vec:
                vector_seq.append(word2vec[word])
            else:
                vector_seq.append(zeros)
        
        if len(vector_seq) < maxlen:
            vector_seq.extend([zeros]*(maxlen-len(vector_seq)))
        return vector_seq

    print('Load Unlabeled Data...', end=' ')
    sys.stdout.flush()
    tStart = time.time()
    nolabel_corpus = load_nolabel_data('../data/training_nolabel.txt', mode='train')
    tEnd = time.time()
    print('cost %s' % tfmt(tEnd-tStart))
    maxlen = max(len(word_seq) for word_seq in nolabel_corpus)
    print('Max Length in Unlabeled Corpus is %d' % maxlen)
    print('Convert Sentences to Vector Sequence...', end=' ')
    sys.stdout.flush()
    tStart = time.time()
    nolabel_x = [padded_vector_sequence(word_seq, 30) for word_seq in nolabel_corpus if len(word_seq) <= 30]
    del nolabel_corpus
    nolabel_x = np.asarray(nolabel_x, dtype='float')
    tEnd = time.time()
    print('cost %s' % tfmt(tEnd-tStart))
    while True:
        a = input('pause$')
