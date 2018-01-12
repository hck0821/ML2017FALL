import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Input, Dot, Add, Flatten, Concatenate
from keras.regularizers import l2

def encoder(input_dim, latent_dim, verbose=1):
    input = Input((input_dim,))
    
    
    hidden = Dense(128, activation='relu')(input)
    hidden = Dense(64, activation='relu')(hidden)
    #hidden = Dense(32, activation='relu')(hidden)
    
    #hidden = Dense(512, activation='relu')(input)
    #hidden = Dropout(0.2)(hidden)
    #hidden = Dense(512, activation='relu')(hidden)
    #hidden = Dropout(0.2)(hidden)
    latent = Dense(latent_dim, activation='tanh')(hidden)
    #latent = Dense(latent_dim)(hidden)

    model = Model(inputs=input, outputs=latent, name='encoder')
    
    if verbose == 1:
        model.summary()
    return model

def decoder(latent_dim, output_dim, verbose=1):
    latent = Input((latent_dim,))
    
    #hidden = Dense(32, activation='relu')(latent)
    hidden = Dense(64, activation='relu')(latent)
    hidden = Dense(128, activation='relu')(hidden)
    
    #hidden = Dense(512, activation='relu')(latent)
    #hidden = Dropout(0.2)(hidden)
    #hidden = Dense(512, activation='relu')(hidden)
    #hidden = Dropout(0.2)(hidden)
    #output = Dense(output_dim)(hidden)
    output = Dense(output_dim, activation='tanh')(hidden)

    model = Model(inputs=latent, outputs=output, name='decoder')

    if verbose == 1:
        model.summary()
    return model

def seq2seq(input_dim, latent_dim, verbose=1):
    input = Input((input_dim,), name='img')
    latent = encoder(input_dim , latent_dim, verbose=0)(input)
    output = decoder(latent_dim, input_dim , verbose=0)(latent)
    model = Model(inputs=input, outputs=output)

    if verbose == 1:
        model.summary()
    return model

if __name__ == '__main__':
    model = seq2seq(784, 32)

