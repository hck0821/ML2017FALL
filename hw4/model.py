import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import GRU, LSTM, Dense, Masking, Dropout, Embedding, Lambda

#maxlen = 41
maxlen = 45
dim = 200

def text_sentiment_classifier(embedding_matrix, cell, verbose=1):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], dim, weights=[embedding_matrix], 
                        input_length=maxlen, trainable=False))
    model.add(Masking(mask_value=0.0))
    
    if cell.upper() == 'GRU':
        #model.add(GRU(256, activation='sigmoid', recurrent_activation='sigmoid', 
        #          dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(GRU(256, activation='sigmoid', recurrent_activation='sigmoid', 
                  dropout=0.2, recurrent_dropout=0.2))
    
    elif cell.upper() == 'LSTM':
        model.add(LSTM(256, activation='sigmoid', recurrent_activation='sigmoid', 
                  dropout=0.2, recurrent_dropout=0.2))
    
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dropout(0.2))
    #model.add(Dense(64, activation='sigmoid'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    if verbose == 1:
        model.summary()
    return model


def bow_dnn(embedding_matrix, verbose=1):
    model = Sequential()
    model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix],
              input_length=maxlen, trainable=False))
    model.add(Lambda(lambda x: K.sum(x,axis=1)))
    model.add(Dense(512, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.add(Dropout(0.2))
    if verbose == 1:
        model.summary()
    return model




if __name__ == '__main__':
    BOW_DNN(20000, verbose=1)
