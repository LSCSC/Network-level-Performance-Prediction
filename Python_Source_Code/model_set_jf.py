from keras.layers import Input, GlobalAveragePooling2D, Reshape, Dense, Flatten, Dropout, LeakyReLU, concatenate
from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda, Activation, Permute
from keras.models import Model, load_model, Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from functools import partial
import keras.backend as K
import tensorflow as tf
import os




#def m_loss(y_true,y_pred):
#
#    x = y_true
#    y = y_pred
#    mx = K.mean(x)
#    my = K.mean(y)
#    xm, ym = x-mx, y-my
#    r_num = K.sum(xm*ym)
#    r_den = K.sqrt(K.sum(K.square(xm)* K.sum(K.square(ym))))
#    r = r_num/r_den
#    diff = K.abs(mx-my)
#    loss = diff*(1-K.square(r))
#    return loss

def m_huber_loss(y_true,y_pred):
    th=0.8
    error = y_true - y_pred
    w = K.abs((y_pred - y_true) / y_true)
    cond = w < th
    square_loss = 0.5 * K.square(error)
    linear_loss = th * y_true * (K.abs(error) - 0.5 * th * y_true)
    return tf.where(cond,square_loss,linear_loss)

def m_loss(y_true,y_pred):
    error = y_true-y_pred
    mse  = K.mean(K.square(error))/200000
    mape = K.mean(K.abs(error)/y_true)
    return mse+mape 

def m_loss2(y_true,y_pred):
    error = K.abs(y_true-y_pred)
    mae  = K.mean(error)/200
    mape = K.mean(error/y_true)
    return mae+mape 

def m_loss3(y_true,y_pred):
    error=K.abs(K.mean(y_true)-K.mean(y_pred))
    return error 

def huber_loss(y_true,y_pred):
    th=30
    error = K.abs(y_true - y_pred)
    cond = error < th
    square_loss = 0.5 * K.square(error)
    linear_loss = th * (K.abs(error) - 0.5 * th )
    return tf.where(cond,square_loss,linear_loss)

def corr(y_true, y_pred):
    mx = K.mean(y_true)
    my = K.mean(y_pred)
    xm, ym = y_true-mx, y_pred-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den
    
    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

def jfmtc(y_true, y_pred):
    error = K.abs(y_true - y_pred)
    r = K.mean(error/(y_pred+y_true+0.001))
    return r

def jf_mse(weights_path=None):

    model = Sequential()

    model.add(ZeroPadding2D((0, 0), input_shape=(20, 268, 1)))

    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(128, (1, 5)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 128)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1,activation='relu'))

    model.summary()

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae','mape'])

    if weights_path!=None:
        model.load_weights(weights_path)
        print('Continue training from ',weights_path)
    else:
        print('Start a new training')

    return model

def jf_mape(weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((0, 0), input_shape=(20, 268, 1)))

    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(128, (1, 5)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 128)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1,activation='relu'))
    model.summary()

    model.compile(optimizer='rmsprop', loss='mape', metrics=['mae','mse'])

    if weights_path!=None:
        model.load_weights(weights_path)
        print('Continue training from ',weights_path)
    else:
        print('Start a new training')

    return model


def jf_sm_mape(weights_path=None):

    model = Sequential()

    model.add(ZeroPadding2D((0, 0), input_shape=(20, 268, 1)))

    model.add(Conv2D(32, (1, 5),padding='same'))

    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))

    model.add(Activation('relu'))

    model.add(Conv2D(32, (1, 5),padding='same'))

    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))

    model.add(Activation('relu'))

    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(64, (1, 5),padding='same'))

    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))

    model.add(Activation('relu'))

    model.add(Conv2D(64, (1, 5),padding='same'))

    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))

    model.add(Activation('relu'))

    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(128, (1, 5),padding='same'))

    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))

    model.add(Activation('relu'))

    model.add(Conv2D(128, (1, 5),padding='same'))

    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))

    model.add(Activation('relu'))

    model.add(Conv2D(128, (1, 5),padding='same'))

    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))

    model.add(Activation('relu'))

    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(256, (1, 5)))

    model.add(Permute((1,3,2)))

    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))

    model.add(Activation('relu'))

    model.add(Conv2D(256, (1, 256)))

    model.add(Permute((1,3,2)))

    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))

    model.add(Activation('relu'))

    model.add(Conv2D(256, (1, 256)))

    model.add(Permute((1,3,2)))

    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))

    model.add(Activation('relu'))
    
    model.add(Flatten())

    model.add(Dense(2048, activation='relu'))

    #model.add(Dense(2048, activation='relu'))

    model.add(Dense(2048, activation='relu'))

    model.add(Dense(1024, activation='relu'))

    model.add(Dense(1024, activation='relu'))

    model.add(Dense(512, activation='relu'))

    model.add(Dense(1))

    model.summary()

    model.compile(optimizer='rmsprop', loss='mape', metrics=['mae','mse'])
    #model.compile(optimizer='rmsprop', loss=m_loss3, metrics=['mse','mae','mape'])

    if weights_path!=None:
        model.load_weights(weights_path)
        print('Continue training from ',weights_path)
    else:
        print('Start a new training')

    return model

def jf_mapev(weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((0, 0), input_shape=(20, 268, 1)))

    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(128, (1, 5)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 128)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer='rmsprop', loss='mape', metrics=['mae','mse'])

    if weights_path!=None:
        model.load_weights(weights_path)
        print('Continue training from ',weights_path)
    else:
        print('Start a new training')

    return model


def jf_mse(weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((0, 0), input_shape=(20, 268, 1)))

    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(128, (1, 5)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 128)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae','mse'])

    if weights_path!=None:
        model.load_weights(weights_path)
        print('Continue training from ',weights_path)
    else:
        print('Start a new training')

    return model


def jf_mae(weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((0, 0), input_shape=(20, 268, 1)))

    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(128, (1, 5)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 128)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer='rmsprop', loss='mae', metrics=['mape','mse'])

    if weights_path!=None:
        model.load_weights(weights_path)
        print('Continue training from ',weights_path)
    else:
        print('Start a new training')

    return model


def jf_ack(weights_path=None):

    model = Sequential()
    model.add(ZeroPadding2D((0, 0), input_shape=(1, 268, 1)))

    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(16, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 5),padding='same'))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((1, 4), strides=(1, 4),padding='same'))#---------------------------------------

    model.add(Conv2D(128, (1, 5)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (1, 128)))
    model.add(Permute((1,3,2)))
    model.add(BatchNormalization(epsilon=1e-5, momentum=0.99))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if weights_path!=None:
        model.load_weights(weights_path)
        print('Continue training from ',weights_path)
    else:
        print('Start a new training')

    return model
