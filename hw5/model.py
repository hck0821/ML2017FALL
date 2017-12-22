import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Embedding, Input, Dot, Add, Flatten, Concatenate
from keras.regularizers import l2

def Matrix_Factorization(num_users, num_movies, latent_size, verbose=1):
    user_input = Input(shape=(1,))
    user_latent = Embedding(num_users+1, latent_size)(user_input)
    user_latent = Dropout(0.4)(user_latent)
    user_latent = Flatten()(user_latent) 

    movie_input = Input(shape=(1,))
    movie_latent = Embedding(num_movies+1, latent_size)(movie_input)
    movie_latent = Dropout(0.4)(movie_latent)
    movie_latent = Flatten()(movie_latent) 

    pred = Dot(axes=1)([user_latent, movie_latent])

    user_bias = Embedding(num_users+1, 1)(user_input)
    user_bias = Flatten()(user_bias)
    
    movie_bias = Embedding(num_movies+1, 1)(movie_input)
    movie_bias = Flatten()(movie_bias)

    pred = Add()([pred, user_bias, movie_bias])

    model = Model(inputs=[user_input, movie_input], outputs=pred)

    if verbose == 1:
        model.summary()
    return model

def DNN(num_users, num_movies, latent_size, verbose=1):
    user_input = Input(shape=(1,))
    user_latent = Embedding(num_users+1, latent_size)(user_input)
    user_latent = Dropout(0.4)(user_latent)
    user_latent = Flatten()(user_latent) 

    movie_input = Input(shape=(1,))
    movie_latent = Embedding(num_movies+1, latent_size)(movie_input)
    movie_latent = Dropout(0.4)(movie_latent)
    movie_latent = Flatten()(movie_latent)

    class_input = Input(shape=(18,))

    pred = Concatenate()([user_latent, movie_latent, class_input])
    pred = Dense(512, activation='relu')(pred)
    pred = Dropout(0.3)(pred)
    pred = Dense(512, activation='relu')(pred)
    pred = Dropout(0.3)(pred)
    pred = Dense(1, activation='relu')(pred)
    model = Model(inputs=[user_input, movie_input, class_input], outputs=pred)

    if verbose == 1:
        model.summary()
    return model
