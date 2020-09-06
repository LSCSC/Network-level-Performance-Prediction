

import os
import TrainTest
import model_set_jf as ms
import numpy as np
import pickle
import copy
import math
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import keras.backend as K 
import tensorflow as tf


def predict_only(path,model_name,pkl_name):

    path_data = os.path.join(path, 'data')
    X_train, y_train, X_test, y_test = TrainTest.prepare_data(path_data, pkl_name)
    TrainTest.predict(path, model_name, X_test, y_test)


def train_only(path, model_name, pkl_name, model_type,snapshot_name,airview_name):

    path_data = os.path.join(path, 'data')
    X_train, y_train, X_test, y_test = TrainTest.prepare_data(path_data,pkl_name,snapshot_name,airview_name,test_ratio=0)
    TrainTest.train(path,model_name,model_type,X_train, y_train,n_epochs=40,n_batch_size=50)

def train_and_predict(path, model_name, pkl_name, model_type,snapshot_name,airview_name):

    path_data = os.path.join(path, 'data')
    X_train, y_train, X_test, y_test = TrainTest.prepare_data(path_data,pkl_name,snapshot_name,airview_name)
    TrainTest.train(path,model_name,model_type,X_train, y_train,n_epochs=40,n_batch_size=50)
    TrainTest.predict(path,model_name, X_test, y_test)

def m_condition(x):
    return ((x<=5)|(x>=15))

def m_roll(X_train,y_train,z,n):
     
    step=int(18/n)
    temp_x=list()
    temp_y=list()
    for index,item in enumerate(X_train):
        if not z[index]==1:
            for ii in range(n):
                x=copy.deepcopy(item[1:])
                x=np.roll(x,step,axis=0)
                item[1:]=x
                temp_x.append(item)
                temp_y.append(y_train[index])
        else:
            temp_x.append(item)
            temp_y.append(y_train[index])
    X_train=np.array(temp_x)
    y_train=np.array(temp_y)

    return X_train,y_train



def m_huber_loss(y_true,y_pred):
    th=0.8
    error = y_true - y_pred
    w = np.abs((y_pred - y_true) / y_true)
    cond = w < th
    square_loss = 0.5 * np.square(error)
    linear_loss = th * y_true * (np.abs(error) - 0.5 * th * y_true)
    return np.mean(np.where(cond,square_loss,linear_loss))

def huber_loss(y_true,y_pred):
    th=30
    error = np.abs(y_true - y_pred)
    cond = error < th
    square_loss = 0.5 * np.square(error)
    linear_loss = th * (np.abs(error) - 0.5 * th )
    return np.where(cond,square_loss,linear_loss)

if __name__ == '__main__':

    path = os.getcwd()
    airview_name_1 = 'result_pop_20s.json'
    airview_name_2 = 'result_pop_20s_2.json'
    airview_name_3 = 'result_pop_20s_3.json'
    airview_name_4 = 'result_pop_20s_4.json'
    airview_name_5 = 'result_pop_20s_5.json'
    airview_name_6 = 'result_pop_20s_6.json'
    airview_name_7 = 'result_pop_20s_7.json'
    airview_name_8 = 'result_pop_20s_8.json'
    airview_name_9 = 'result_pop_20s_9.json'
    airview_name_10 = 'result_pop_20s_10.json'
    airview_name_11 = 'result_pop_20s_11.json'
    airview_name_12 = 'result_pop_20s_12.json'
    airview_name_13 = 'result_pop_20s_13.json'


    snapshot_name_1 = 'snapshot_dr.json'
    snapshot_name_2 = 'snapshot_dr_2.json'
    snapshot_name_3 = 'snapshot_dr_3.json'
    snapshot_name_4 = 'snapshot_dr_4.json'
    snapshot_name_5 = 'snapshot_dr_5.json'
    snapshot_name_6 = 'snapshot_dr_6.json'
    snapshot_name_7 = 'snapshot_dr_7.json'
    snapshot_name_8 = 'snapshot_dr_8.json'
    snapshot_name_9 = 'snapshot_dr_9.json'
    snapshot_name_10 = 'snapshot_dr_10.json'
    snapshot_name_11 = 'snapshot_dr_11.json'
    snapshot_name_12 = 'snapshot_dr_12.json'
    snapshot_name_13 = 'snapshot_dr_13.json'
    pkl_name_1 = 'snapshot_dr.pkl'
    pkl_name_2 = 'snapshot_dr_2.pkl'
    pkl_name_3 = 'snapshot_dr_3.pkl'
    pkl_name_4 = 'snapshot_dr_4.pkl'
    pkl_name_5 = 'snapshot_dr_5.pkl'
    pkl_name_6 = 'snapshot_dr_6.pkl'
    pkl_name_7 = 'snapshot_dr_7.pkl'
    pkl_name_8 = 'snapshot_dr_8.pkl'
    pkl_name_9 = 'snapshot_dr_9.pkl'
    pkl_name_10 = 'snapshot_dr_10.pkl'
    pkl_name_11 = 'snapshot_dr_11.pkl'
    pkl_name_12 = 'snapshot_dr_12.pkl'
    pkl_name_13 = 'snapshot_dr_13.pkl'


    test_set_name = 'test_set.pkl'
    std_name = 'norm_standard.pkl'

    #model_type = ms.c_structure

    path_data = os.path.join(path, 'data')
    for ii in range(1,14):
        exec('X_train_%s, _, a_train_%s, X_test_%s, _, a_test_%s, z_%s = TrainTest.prepare_data(path_data,pkl_name_%s,snapshot_name_%s,airview_name_%s,std_name,test_ratio=0)'%(ii,ii,ii,ii,ii,ii,ii,ii))




    for ii in range(1,14):
        exec('X_train_%s = X_train_%s[:,0,:,:]'%(ii,ii))
        exec('X_train_%s = X_train_%s.reshape((X_train_%s.shape[0],1,X_train_%s.shape[1], X_train_%s.shape[2]))'%(ii,ii,ii,ii,ii))
        exec('X_train_%s[:,:,256:,:]=-1'%ii)


    X_train=np.concatenate((X_train_1,X_train_2,X_train_3,X_train_4,X_train_5,X_train_6,X_train_7,X_train_8,X_train_9,X_train_10,X_train_11,X_train_12,X_train_13),axis=0)
    a_train=np.concatenate((a_train_1,a_train_2,a_train_3,a_train_4,a_train_5,a_train_6,a_train_7,a_train_8,a_train_9,a_train_10,a_train_11,a_train_12,a_train_13),axis=0)



    for ii in range(1,14):
        exec('X_train_%s = []'%ii)
        exec('a_train_%s = []'%ii)
        exec('z_%s = []'%ii)
    #offset=-int(np.mean(y_train))
    np.random.seed(1)
    np.random.shuffle(X_train)
    np.random.seed(1)
    np.random.shuffle(a_train)
    #y_train[y_train<0.1]=0
    #y_test[y_test<0.2]=0
    thre1=int(0.6*len(a_train))
    thre2=int(0.8*len(a_train))
    X_test=X_train[thre2:]
    a_test=a_train[thre2:]
    X_val=X_train[thre1:thre2]
    a_val=a_train[thre1:thre2]
    X_train=X_train[:thre1]
    a_train=a_train[:thre1]

    print('done shuffling %s'%ii)



    ne=100
    loss=float('inf')
    oname='njf_same_ack_random'
    model_name = oname+'.h5'
    model_type = ms.jf_ack
    model = model_type()
    path_model = os.path.join(path, 'model',model_name)


    a_integers = np.argmax(a_train, axis=1)
    c_weights = compute_class_weight('balanced', np.unique(a_integers), a_integers)
    class_weights = dict(enumerate(c_weights))
    history=TrainTest.train(path,model_name,model_type,X_train, a_train,n_epochs=40,n_batch_size=256,class_weight=class_weights,vls=(X_val,a_val))

    hist_name = os.path.join(path_data, 'hist_'+oname+'.pkl')
    with open(hist_name, 'wb') as df:
        pickle.dump((history.history['val_loss'],history.history['val_acc'],history.history['loss'],history.history['acc']), df)

    path_data = os.path.join(path, 'data/result_'+oname+'.pkl')
    TrainTest.predict(path,model_name,X_test,a_test,path_data)


