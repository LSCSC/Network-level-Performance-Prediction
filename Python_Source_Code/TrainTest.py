from keras.models import Model, load_model, Sequential
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from model_set import huber_loss, m_huber_loss
import keras.backend as K
import os
import model_set as ms
import numpy as np
#import snapshot_tti_individual_dr as sti
import pickle
import matplotlib.pyplot as plt
import copy


def prepare_data(path,pkl_name,snapshot_name=None,airview_name=None,std_name='std',test_ratio=0.2,pre_only=False):
    pkl_name = os.path.join(path, pkl_name) #---------------------------------------------------------------
    std_name = os.path.join(path, std_name)
    try:
        with open(pkl_name, 'rb') as df:
            x,y,a,z = pickle.load(df)
    except FileNotFoundError:
#        infos = sti.load_from_json(path,snapshot_name,airview_name)
#        x = infos['features']
#        y = infos['rates']
#        a = infos['acks']

        if pre_only:
            x,z = zero_paddingx(x,h=20)
            return x
        else:
            with open(std_name, 'rb') as df:
                min_list,max_list,minus_ue = pickle.load(df)

            x,z = zero_paddingx(x,p=minus_ue,h=20)

        for j in range(x.shape[2]):
            x[:, :, j] = x[:, :, j] - min_list[j]
            if max_list[j] > 0:
                x[:, :, j] = x[:, :, j] / max_list[j]

        with open(pkl_name, 'wb') as df:
            pickle.dump((x,y,a,z), df, protocol=4)



    #np.random.seed(100)
    #np.random.shuffle(x)
    #np.random.seed(100)
    #np.random.shuffle(y)
    size = len(x)
    t1=0.5-test_ratio/2
    t2=0.5+test_ratio/2
    t1=int(size * t1)
    t2=int(size * t2)
    X_train = np.concatenate((x[:t1],x[t2:]),axis=0)
    X_test = x[t1:t2]
    y_train = y[:t1]+y[t2:]
    y_test = y[t1:t2]
    a_train = a[:t1]+a[t2:]
    a_test = a[t1:t2]


    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    # print(X_train.shape)
    y_train = np.array(y_train).reshape(len(y_train)) / (10 ** 4)
    a_train = np.array(a_train).reshape(len(a_train),4)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    y_test = np.array(y_test).reshape(len(y_test)) / (10 ** 4)
    a_test = np.array(a_test).reshape(len(a_test),4)



    print('done preparing, load pkl from', pkl_name)
    return X_train, y_train, a_train, X_test, y_test, a_test,z



def zero_paddingx(x,p=None,h=20):
    l_pad=len(x[0][0])
    z=list()
    yy=list()
    for i_index, ii in enumerate(x):
        #print(i_index,'/',len(x))
        for index, item in enumerate(ii):
            ii[index] = np.reshape(item, (1, l_pad))
        ii = np.concatenate(ii, axis=0)
        yy.append(ii.shape[0])
        pad_len = h - int(ii.shape[0])
        #assert(pad_len==0)
        if p is None:
           p=copy.deepcopy(ii[0,:])
    
        if pad_len > 0:
            z.append(np.concatenate((ii, np.tile(p,(pad_len,1))), axis=0))
        else:
            z.append(ii[:h])
    zz = np.stack(z, axis=0)
    return zz,yy

def make_std(x,path,std_name):
    std_name = os.path.join(path, std_name)
    min_list = x.min(axis=0).min(axis=0)
    max_list = x.max(axis=0).max(axis=0)-min_list

    minus_ue=np.zeros(min_list.shape)
    for index,item in enumerate(min_list):
        if max_list[index]!=0:
            minus_ue[index]=-max_list[index]+min_list[index]
        else:
            minus_ue[index]=min_list[index]-1

    with open(std_name, 'wb') as df:
        pickle.dump((min_list, max_list,minus_ue), df)



def train(path,model_name,model_type,X_train,y_train,class_weight=None,base_model=None,n_epochs=20,n_batch_size=50,patience=40,vls=None):
    path_model = os.path.join(path, 'model',model_name) #---------------------------------------------------------------
    if base_model!=None:
        base_model = os.path.join(path, 'model',base_model)
        model = model_type(base_model)
    else:
        model = model_type()
    history=model.fit(X_train,y_train, epochs=n_epochs, batch_size=n_batch_size, verbose=1,class_weight=class_weight,validation_data=vls)
    model.save(path_model, overwrite=True)
    return history

def predict(path,model_name, X_test, y_test,save_name):
    path_data = os.path.join(path, save_name)
    path_model = os.path.join(path, 'model',model_name) #---------------------------------------------------------------
    # X_train, y_train, X_test, y_test = prepare_data(path_data)
    # print(X_test)
    #model = load_model(path_model,custom_objects={'m_huber_loss': m_huber_loss})
    model = load_model(path_model)
    model.summary()
    # model1 = Model(inputs= model.input, outputs= model.get_layer('conv2d_8').output)
    # predicts = model1.predict(X_test)
    predicts = model.predict(X_test)
    #print('ss',type(predicts))
    with open(path_data, 'wb') as df:
        pickle.dump((predicts, y_test), df)

def predict_mcs(path,model_name, X_test, y_test, c_mcs, save_name):
    path_data = os.path.join(path, save_name)
    path_model = os.path.join(path, 'model',model_name) #---------------------------------------------------------------
    # X_train, y_train, X_test, y_test = prepare_data(path_data)
    # print(X_test)
    model = load_model(path_model,custom_objects={'m_huber_loss': m_huber_loss})
    #model = load_model(path_model)
    #model.summary()
    # model1 = Model(inputs= model.input, outputs= model.get_layer('conv2d_8').output)
    # predicts = model1.predict(X_test)
    predicts = model.predict(X_test)
    #print('ss',type(predicts))
    with open(path_data, 'wb') as df:
        pickle.dump((predicts, y_test,c_mcs), df)


if __name__ == '__main__':
    path = os.getcwd()
    path_data = os.path.join(path, 'data')
    airview_name = 'result_pop_20s.json'
    snapshot_name = 'snapshot_dr.json'
    model_name = 'vgg_dr.h5'
    pkl_name = 'snapshot_dr.pkl'
    X_train, y_train, X_test, y_test = prepare_data(path_data,pkl_name,snapshot_name,airview_name)
    model_type = VGG_16
    #train(path,model_name,model_type,X_train, y_train,n_epochs=20,n_batch_size=50)
    predict(path,model_name, X_test, y_test)


