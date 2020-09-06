

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
from sklearn.metrics import confusion_matrix


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
    oname='njf_tl_ack_random'
    path_data = os.path.join(path, 'data/result_'+oname+'.pkl')
    with open(path_data, 'rb') as df:
        predicts, y_test= pickle.load(df)
    for item in predicts:
        item[:]=item==item.max()
    #C=confusion_matrix(y_test,predicts)
    #print(type(predicts))
    #print(predicts.shape)
    #print(predicts[1]  y_test[1])
    #print(predicts[1])
    #print(y_test[1])
    #print(C)
    C= np.sum(y_test,axis=0)
    print(C)
    print(np.sum(predicts[y_test[:,0]==1],axis=0))
    print(np.sum(predicts[y_test[:,1]==1],axis=0))
    print(np.sum(predicts[y_test[:,2]==1],axis=0))
    print(np.sum(predicts[y_test[:,3]==1],axis=0))
    print('----------------------------------')
    print(np.sum(predicts[y_test[:,0]==1],axis=0)/C[0]*100)
    print(np.sum(predicts[y_test[:,1]==1],axis=0)/C[1]*100)
    print(np.sum(predicts[y_test[:,2]==1],axis=0)/C[2]*100)
    print(np.sum(predicts[y_test[:,3]==1],axis=0)/C[3]*100)


