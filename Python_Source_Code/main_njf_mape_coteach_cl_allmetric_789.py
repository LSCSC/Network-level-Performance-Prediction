

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
from keras.models import Model, load_model, Sequential
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

def layer_split(X_train):
    l2_1=X_train[:,:,:38,:]
    l2_2=X_train[:,:,[40,41,44,45,47],:]
    l2_3=X_train[:,:,133:150,:]
    l2_4=X_train[:,:,256:,:]
    l2=np.concatenate((l2_1,l2_2,l2_3,l2_4),axis=2)
    l_cqi=X_train[:,:,154:254,:]
    l1_1=X_train[:,:,[38,39,42,43,46],:]
    l1_2=X_train[:,:,48:133,:]
    l1_3=X_train[:,:,[150,151,152,153,254,255],:]
    l1=np.concatenate((l1_1,l1_2,l1_3),axis=2)
    return l1,l2,l_cqi

def m_huber_loss(y_true,y_pred):
    th=0.8
    error = y_true - y_pred
    w = np.abs((y_pred - y_true) / y_true)
    cond = w < th
    square_loss = 0.5 * np.square(error)
    linear_loss = th * y_true * (np.abs(error) - 0.5 * th * y_true)
    return np.mean(np.where(cond,square_loss,linear_loss))

def mape(y_true,y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))

def mape_loss(y_true,y_pred):
    return np.abs((y_pred-y_true)/y_true)


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
        exec('X_train_%s, y_train_%s, _, X_test_%s, y_test_%s, _, z_%s = TrainTest.prepare_data(path_data,pkl_name_%s,snapshot_name_%s,airview_name_%s,std_name,test_ratio=0)'%(ii,ii,ii,ii,ii,ii,ii,ii))


    for ii in range(1,14):
        np.random.seed(1)
        exec('np.random.shuffle(X_train_%s)'%ii)
        np.random.seed(1)
        exec('np.random.shuffle(y_train_%s)'%ii)
        np.random.seed(1)
        exec('np.random.shuffle(z_%s)'%ii)

    for ii in range(1,14):
        exec('for item in X_train_%s:x=copy.deepcopy(item[1:]);np.random.shuffle(x);item[1:]=x'%ii)
    


    print('done shuffling %s'%ii)


    X_test=np.concatenate((X_train_7[:6000],X_train_8[:6000],X_train_9[:6000]),axis=0)
    #test_l1,test_l2,test_l_cqi=layer_split(X_test)
    y_test=np.concatenate((y_train_7[:6000],y_train_8[:6000],y_train_9[:6000]),axis=0)
    X_train=np.concatenate((X_train_1,X_train_2,X_train_3,X_train_4,X_train_5,X_train_6,X_train_10,X_train_11,X_train_12,X_train_13),axis=0)
    y_train=np.concatenate((y_train_1,y_train_2,y_train_3,y_train_4,y_train_5,y_train_6,y_train_10,y_train_11,y_train_12,y_train_13),axis=0)

    #X_train=X_train_1
    #y_train=y_train_1
    #X_train_1 = []
    #y_train_1 = []
    #tlist=[2,3,4,5,6,10,11,12,13]
    #for ii in tlist:
    #    exec('X_train = np.concatenate((X_train,X_train_%s),axis=0)'%ii)
    #    exec('y_train = np.concatenate((y_train,y_train_%s),axis=0)'%ii)
    #    exec('X_train_%s = []'%ii)
    #    exec('y_train_%s = []'%ii)
    #X_test=np.concatenate((X_train_1[:6000],X_train_3[:6000],X_train_5[:6000]),axis=0)
    ##test_l1,test_l2,test_l_cqi=layer_split(X_test)
    #y_test=np.concatenate((y_train_1[:6000],y_train_3[:6000],y_train_5[:6000]),axis=0)
    #X_train=np.concatenate((X_train_2,X_train_4,X_train_6,X_train_7,X_train_8,X_train_9),axis=0)
    #y_train=np.concatenate((y_train_2,y_train_4,y_train_6,y_train_7,y_train_8,y_train_9),axis=0)

    for ii in range(1,14):
        exec('X_train_%s = []'%ii)
        exec('y_train_%s = []'%ii)

    #offset=-int(np.mean(y_train))
    #y_train[y_train<0.1]=0
    #y_test[y_test<0.2]=0
    offset=0.5
    print('offset',offset)
    y_train=y_train+offset
    y_test=y_test+offset


    #np.random.seed(1)
    #np.random.shuffle(X_train)
    #np.random.seed(1)
    #np.random.shuffle(y_train)
    #l1,l2,l_cqi=layer_split(X_train)


    for tl in range(25,50):
        record_metrics2=list()
        loss=float('inf')
        model_type = ms.jf_mapev
        exec('oname="njf_cl_coteach_weight_mape_allmetric_alg_789_%s"%tl')
        #oname='njf_cl_coteach_mape_allmetric_alg_789'
        model_name_A = oname+'_A.h5'
        model_name_B = oname+'_B.h5'
        path_model_A = os.path.join(path, 'model',model_name_A)
        path_model_B = os.path.join(path, 'model',model_name_B)
        model_A = model_type()
        model_B = model_type()

        #history=model.fit([l1,l2,l_cqi],y_train, epochs=1, batch_size=64, verbose=1)
        np.random.seed(83)
        model_A.fit(X_train,y_train, epochs=1, batch_size=256, verbose=1)
        model_B.fit(X_train,y_train, epochs=1, batch_size=256, verbose=1)
        bs=int(len(y_train)/10)
        ip=0
        ne=6
        for ii in range(ne):
            if (ii-ip)<160:
                print('iteration: %d'%ii)
  

                predicts = (model_A.predict(X_test).flatten()+model_B.predict(X_test).flatten())/2
                predicts[predicts<offset]=offset
                mse=round(np.mean(np.square(predicts-y_test)),3)
                error=predicts[:6000]-y_test[:6000]
                md7=round(np.mean(np.abs((error)/(y_test[:6000]-offset+5))),3)
                me7=round(np.max(np.abs(error)),3)
                ab7=round(np.mean((2*error/(y_test[:6000]+predicts[:6000]-2*offset+0.001))),3)
                ac7=round(np.mean(np.abs(error/(y_test[:6000]+predicts[:6000]+0.001))),3)
                cr7=round(np.corrcoef(predicts[:6000],y_test[:6000])[0,1],3)
                mv7=round(np.mean(predicts[:6000])-np.mean(y_test[:6000]),3)
                nv7=round(mv7/np.mean(y_test[:6000]-offset),3)
                aa7=round(np.mean(np.abs(error/(predicts[:6000]+y_test[:6000]-2*offset+0.001))),3)
	    		
                error=predicts[6000:12000]-y_test[6000:12000]
                md8=round(np.mean(np.abs((error)/(y_test[6000:12000]-offset+5))),3)
                me8=round(np.max(np.abs(error)),3)
                ab8=round(np.mean((2*error/(y_test[6000:12000]+predicts[6000:12000]-2*offset+0.001))),3)
                ac8=round(np.mean(np.abs(error/(y_test[6000:12000]+predicts[6000:12000]+0.001))),3)
                cr8=round(np.corrcoef(predicts[6000:12000],y_test[6000:12000])[0,1],3)
                mv8=round(np.mean(predicts[6000:12000])-np.mean(y_test[6000:12000]),3)
                nv8=round(mv8/np.mean(y_test[6000:12000]-offset),3)
                aa8=round(np.mean(np.abs(error/(predicts[6000:12000]+y_test[6000:12000]-2*offset+0.001))),3)

                error=predicts[12000:]-y_test[12000:]
                md9=round(np.mean(np.abs((error)/(y_test[12000:]-offset+5))),3)
                me9=round(np.max(np.abs(error)),3)
                ab9=round(np.mean((2*error/(y_test[12000:]+predicts[12000:]-2*offset+0.001))),3)
                ac9=round(np.mean(np.abs(error/(y_test[12000:]+predicts[12000:]+0.001))),3)
                cr9=round(np.corrcoef(predicts[12000:],y_test[12000:])[0,1],3)
                mv9=round(np.mean(predicts[12000:])-np.mean(y_test[12000:]),3)
                nv9=round(mv9/np.mean(y_test[12000:]-offset),3)
                aa9=round(np.mean(np.abs(error/(predicts[12000:]+y_test[12000:]-2*offset+0.001))),3)
	    		
                error=predicts-y_test
                ab=round(np.mean((2*error/(y_test+predicts-2*offset+0.001))),3)
                aa=round(np.mean(np.abs(error/(y_test+predicts-2*offset+0.001))),3)
                ac=round(np.mean(np.abs(error/(y_test+predicts+0.001))),3)
	    		
                md =round(np.mean(np.abs((error)/(y_test-offset+5))),3)
                mv=round(np.mean(predicts)-np.mean(y_test),3)
                nv=round(mv/np.mean(y_test-offset),3)
                cr= round(np.corrcoef(predicts,y_test)[0,1],3)
                md=[md7,md8,md9,md]
                mv=[mv7,mv8,mv9,mv]
                nv=[nv7,nv8,nv9,nv]
                cr=[cr7,cr8,cr9,cr]
                me=[me7,me8,me9]
                aa=[aa7,aa8,aa9,aa]
                ab=[ab7,ab8,ab9,ab]
                ac=[ac7,ac8,ac9,ac]
                print('relative_diffence_test:',md,'mse:',mse,'mean_diffence:',mv,'relative_mean_difference:',nv,'maximum_difference',me,'corr:',cr,'metric_houchao:',aa,'metric_prefer:',ab,'metric_houchao+1:',ac)
                record_metrics2.append([md,mse,nv,cr,aa,ab,ac])


                for item in X_train:
                    x=copy.deepcopy(item[1:])
                    np.random.shuffle(x)
                    item[1:]=x       

                np.random.seed(ii)
                np.random.shuffle(X_train)
                np.random.seed(ii)
                np.random.shuffle(y_train)
                #predicts = model.predict([test_l1,test_l2,test_l_cqi])
                #p_offset=offset
                #offset=np.maximum(offset*0.6,0.1)
                #print('offset',offset)
                #y_train=y_train+offset-p_offset
                #y_test=y_test+offset-p_offset

                if ii<(ne-1):
                    for idx in range(10):
                        temp_train=X_train[idx*bs:(idx+1)*bs]
                        temp_label=y_train[idx*bs:(idx+1)*bs]
                        
                        predicts = model_A.predict(temp_train).flatten()
                        w = np.abs((predicts-temp_label)/temp_label)
                        w = np.maximum(1-w,0)
                        model_B.fit(temp_train,temp_label, epochs=1, batch_size=64, verbose=0,sample_weight=w)
                        
                        predicts = model_B.predict(temp_train).flatten()
                        w = np.abs((predicts-temp_label)/temp_label)
                        w = np.maximum(1-w,0)
                        model_A.fit(temp_train,temp_label, epochs=1, batch_size=64, verbose=0,sample_weight=w)

                #if ii==6:
                #    model_A.save(path_model_A, overwrite=True)
                #    model_B.save(path_model_B, overwrite=True)

            else:
                break

        #model_name = 'sm_dr_1_to_6_m_huber_08_200.h5'
        #TrainTest.predict(path,model_name,X_test,y_test,'data/result_cs_80_l12.pkl')
        #model = load_model(path_model,custom_objects={'m_huber_loss': ms.m_huber_loss})

    #model = load_model(path_model)

    #model_A = load_model(path_model_A)
    #model_B = load_model(path_model_B)
    #predicts = (model_A.predict(X_test).flatten()+model_B.predict(X_test).flatten())/2
    #predicts = model.predict([test_l1,test_l2,test_l_cqi])

    #predicts = model.predict(X_test)
        hist_name = os.path.join(path_data, 'hist_'+oname+'.pkl')
        with open(hist_name, 'wb') as df:
            pickle.dump(record_metrics2, df)


    #model = load_model(path_model)

    #model = load_model(path_model,custom_objects={'m_loss3': ms.m_loss3})
    #predicts = model.predict([test_l1,test_l2,test_l_cqi])

    #predicts = model.predict(X_test)
    #path_data = os.path.join(path, 'data/result_'+oname+'.pkl')
    #with open(path_data, 'wb') as df:
    #    pickle.dump((predicts, y_test), df)
    
	
#    X_test=np.concatenate((X_train_7[:6000],X_train_8[:6000],X_train_9[:6000]),axis=0)
#    y_test=np.concatenate((y_train_7[:6000],y_train_8[:6000],y_train_9[:6000]),axis=0)
#    X_test[:,:,256:,:]=-1
#    X_test[:,:,:38,:]=-1
#    X_test[:,:,133:150,:]=-1
#    X_test[:,:,[40,41,44,45,266,267,47,259],:]=-1
#    test_l1,test_l2,test_l_cqi=layer_split(X_test)
#
#    ne=120
#    record_metrics1=list()
#    loss=float('inf')
#    model_name = 'cs_120_l1.h5'
#    model = model_type()
#    path_model = os.path.join(path, 'model',model_name)
#    X_train=np.concatenate((X_train_1,X_train_2,X_train_3,X_train_4,X_train_5,X_train_6),axis=0)
#    y_train=np.concatenate((y_train_1,y_train_2,y_train_3,y_train_4,y_train_5,y_train_6),axis=0)
#    np.random.seed(1)
#    np.random.shuffle(X_train)
#    np.random.seed(1)
#    np.random.shuffle(y_train)
#
#    X_train[:,:,256:,:]=-1
#    X_train[:,:,:38,:]=-1
#    X_train[:,:,133:150,:]=-1
#    X_train[:,:,[40,41,44,45,266,267,47,259],:]=-1
#    l1,l2,l_cqi=layer_split(X_train)
#
#    history=model.fit([l1,l2,l_cqi],y_train, epochs=1, batch_size=64, verbose=1)
#
#    ip=0
#    np.random.seed(82)
#    for ii in range(ne):
#        if (ii-ip)<160:
#            print('iteration: %d'%ii)
#
#            for item in X_train:
#                x=copy.deepcopy(item[1:])
#                np.random.shuffle(x)
#                item[1:]=x       
#
#            predicts = model.predict([test_l1,test_l2,test_l_cqi])
#            predicts = predicts.flatten()
#            md = np.mean(np.abs((predicts-y_test)/y_test))
#            mv=np.mean(predicts)-np.mean(y_test)
#            mse=np.mean(np.square(predicts-y_test))
#            nv=mv/np.mean(y_test-10)
#            print('relative_diffence_test:',md,'mean_diffence:',mv,'relative_mean_difference:',nv)
#            record_metrics1.append([md,mse,mv,nv])
#
#            l1,l2,l_cqi=layer_split(X_train)
#            predicts = model.predict([l1,l2,l_cqi])
#            predicts = predicts.flatten()
#            md = np.mean(m_huber_loss(y_train,predicts))
#            print('m_loss:',md)
#            if loss>md:
#                print('performance is improved from %f to %f'%(loss,md))
#                loss=md
#                ip=ii
#                model.save(path_model, overwrite=True)
#            
#            history=model.fit([l1,l2,l_cqi],y_train, epochs=1, batch_size=64, verbose=1)
#            #w = np.abs((predicts-y_train)/y_train)/3
#            #w = np.maximum(1-w,0)
#            #print('%f of all data is used'%(len(w[(1-w)>0])/len(w)))
#            #history=model.fit(X_train,y_train, epochs=1, batch_size=64, verbose=1,sample_weight=w)
#        else:
#            break
#
#    #model_name = 'sm_dr_1_to_6_m_huber_08_200.h5'
#    #TrainTest.predict(path,model_name,X_test,y_test,'data/result_1_to_6_m_huber_08_200_l1.pkl')
#
#
#    
#    hist_name = os.path.join(path_data, 'hist.pkl')
#    with open(hist_name, 'wb') as df:
#        pickle.dump((record_metrics1, record_metrics2), df, protocol=4)
