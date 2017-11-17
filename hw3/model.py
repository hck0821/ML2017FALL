from __future__ import print_function
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.layers.merge import concatenate
import utils

n_classes = 7

def CNN():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(48,48,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation='softmax'))

    model.summary()
    return model


def VGG10():
    model = Sequential()
    
    # Convolution block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
    #model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # Convolution block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # Convolution block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    
    
    # Classifier block
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(n_classes, activation='softmax'))

    model.summary()
    return model

def Conv2D_base(x, filters, kernel_size, strides=(1, 1), padding='same', bias=False):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=bias)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def inception_a(input):
    x1 = Conv2D_base(input, 96, (1, 1))

    x2 = Conv2D_base(input, 64, (1, 1))
    x2 = Conv2D_base(x2, 96, (3, 3))

    x3 = Conv2D_base(input, 64, (1, 1))
    x3 = Conv2D_base(x3, 96, (3, 3))
    x3 = Conv2D_base(x3, 96, (3, 3))
    
    x4 = AveragePooling2D((3, 3), strides=(1, 1),  padding='same')(input)
    x4 = Conv2D_base(x4, 96, (1, 1))

    x = concatenate([x1, x2, x3, x4])
    return x

def reduction_a(input):
    r1 = Conv2D_base(input, 384, (3, 3), strides=(2, 2), padding='valid')

    r2 = Conv2D_base(input, 192, (1, 1))
    r2 = Conv2D_base(r2, 224, (3, 3))
    r2 = Conv2D_base(r2, 256, (3, 3), strides=(2, 2), padding='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    r = concatenate([r1, r2, r3])
    return r

def inception_b(input):
    x1 = Conv2D_base(input, 384, (1, 1))

    x2 = Conv2D_base(input, 192, (1, 1))
    x2 = Conv2D_base(x2, 224, (1, 5))
    x3 = Conv2D_base(x2, 256, (5, 1))

    x3 = Conv2D_base(input, 192, (1, 1))
    x3 = Conv2D_base(x3, 192, (5, 1))
    x3 = Conv2D_base(x3, 224, (1, 5))
    x3 = Conv2D_base(x3, 224, (5, 1))
    x3 = Conv2D_base(x3, 256, (1, 5))

    x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    x4 = Conv2D_base(x4, 128, (1, 1))

    x = concatenate([x1, x2, x3, x4])
    return x

def reduction_b(input):
    r1 = Conv2D_base(input, 192, (1, 1))
    r1 = Conv2D_base(r1, 192, (3, 3), strides=(2, 2), padding='valid')

    r2 = Conv2D_base(input, 256, (1, 1))
    r2 = Conv2D_base(r2, 256, (1, 5))
    r2 = Conv2D_base(r2, 320, (5, 1))
    r2 = Conv2D_base(r2, 320, (3, 3), strides=(2, 2), padding='valid')

    r3 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    r = concatenate([r1, r2, r3])
    return r

def inception_c(input):
    x1 = Conv2D_base(input, 256, (1, 1))

    x2 = Conv2D_base(input, 384, (1, 1))
    x21 = Conv2D_base(x2, 256, (1, 3))
    x22 = Conv2D_base(x2, 256, (3, 1))
    x2 = concatenate([x21, x22])

    x3 = Conv2D_base(input, 384, (1, 1))
    x3 = Conv2D_base(x3, 448, (3, 1))
    x3 = Conv2D_base(x3, 512, (1, 3))
    x31 = Conv2D_base(x3, 256, (1, 3))
    x32 = Conv2D_base(x3, 256, (3, 1))
    x3 = concatenate([x31, x32])

    x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    x4 = Conv2D_base(x4, 256, (1, 1))

    x = concatenate([x1, x2, x3, x4])
    return x

def inception():
    inputs = Input((48, 48, 1))
    
    x = Conv2D_base(inputs, 64, (3, 3), strides=(2, 2), padding='valid')
    x = Conv2D_base(x, 64, (3, 3))
    x = Conv2D_base(x, 128, (3, 3))

    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x2 = Conv2D_base(x, 256, (3, 3), strides=(2, 2), padding='valid')

    x = concatenate([x1, x2])
    x = Dropout(0.3)(x)

    x1 = Conv2D_base(x, 128, (1, 1))
    x1 = Conv2D_base(x1, 256, (3, 3), padding='valid')

    x2 = Conv2D_base(x, 128, (1, 1))
    x2 = Conv2D_base(x2, 128, (1, 5))
    x2 = Conv2D_base(x2, 128, (5, 1))
    x2 = Conv2D_base(x2, 256, (3, 3), padding='valid')

    x = concatenate([x1, x2])
    x = Dropout(0.3)(x)

    x1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)
    x2 = Conv2D_base(x, 512, (3, 3), strides=(2, 2), padding='valid')
    
    x = concatenate([x1, x2])
    x = Dropout(0.3)(x)
    
    '''    
    x = inception_a(x)
    x = reduction_a(x)
    x = inception_b(x)
    x = reduction_b(x)
    x = inception_c(x)
    '''

    x = Flatten()(x)
    x = Dense(units=512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(units=n_classes, activation='softmax')(x)

    model = Model(inputs, x)
    model.summary()

    return model

'DNN for report'
def DNN():
    model = Sequential()
    ''' 
    model.add(Dense(4096, activation='relu', input_shape=(48*48,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    '''
    model.add(Dense(2048, activation='relu', input_shape=(48*48,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    model.add(Dense(n_classes, activation='relu'))
    
    model.summary()
    return model
