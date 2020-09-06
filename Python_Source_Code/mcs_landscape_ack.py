

import os
import TrainTest 
import model_set as ms
import numpy as np
import pickle
import copy
from keras.models import Model, load_model, Sequential
import matplotlib.pyplot as plt
#import snapshot_order as so



if __name__ == '__main__':
    test_set_name = 'test_set.pkl'
    std_name = 'norm_standard.pkl'
    
    path = os.getcwd()
    path_data = os.path.join(path, 'data')

    std_name = os.path.join(path_data, std_name)


    with open(std_name, 'rb') as df:
        x,y,z = pickle.load(df)
    
    test_set_name = os.path.join(path_data, 'test_set.pkl')
    with open(test_set_name, 'rb') as df:
        X_test,y_test,a_test = pickle.load(df)
    
    
    oname='njf_tl_ack_random'
    model_name = oname+'.h5'
    path_model = os.path.join(path, 'model',model_name)
    model = load_model(path_model)


    sample_index=np.random.randint(len(X_test))
    mcs_l=list()
    se_l=list()
    for mcss in range(1,30):
        print(mcss)
        temp=0
        if mcss==5:
            se_l.append(temp)
            continue
        for jj in range(10):
            print('-----------------------',jj)
            while not X_test[sample_index,0,39,0]*y[39]+x[39]==mcss:
                sample_index=np.random.randint(len(X_test))
            mcs=X_test[sample_index,0,[39,43],0]*y[39]+x[39]
            tb_size=X_test[sample_index,0,[40,44],0]*y[40]+x[40]
            tb_pw=X_test[sample_index,0,49:99,0]*y[49]+x[49]
            sample_index=np.random.randint(len(X_test))
            if temp<tb_size[0]/np.sum(tb_pw>0):
                temp=tb_size[0]/np.sum(tb_pw>0)
        mcs_l.append(mcs[0])
        se_l.append(temp)

    se_l[4]=(se_l[3]+se_l[5])/2


    xx = np.arange(1,30)
    yy = np.array(se_l)
    z1 = np.polyfit(xx, yy, 4)
    p1 = np.poly1d(z1)
    se_l=p1(xx)


    c_pre=list()
    c_cmcs=list()
    a_s=list()
    ef_tb=list()
    for sample_index in range(len(X_test)):
        if X_test[sample_index][0,43,0]>0:
            print(sample_index)
            continue 
        coll=list()
        test_label=np.array([y_test[sample_index]for i in range(1,30)])
        c_mcs=X_test[sample_index][0,39,0]*y[39]+x[39]
        rb_num=np.sum(X_test[sample_index,0,49:99,0]*y[49]+x[49]>0)
        tb_set=np.empty(shape=[1,0])
        for ii in range(1,30):
            temp=copy.deepcopy(X_test[sample_index])
            temp[0,39,0]=(ii-x[39])/y[39]
            buffer_size = temp[0,2,0]*y[2]+x[2]
            tb_size_pre=np.min([se_l[ii-1]*rb_num,buffer_size])
            tb_set=np.append(tb_set,tb_size_pre)
            temp[0,40,0]=(tb_size_pre-x[40])/y[40]
            #temp[0,40,0]=np.max([(se_l[ii-1]*rb_num-x[40])/y[40]])
            temp[0,256:,0]=-1
            coll.append(temp)
        test_sample=np.stack(coll,axis=0)
        

        test_sample = test_sample[:,0,:,:]
        test_sample = test_sample.reshape((test_sample.shape[0],1,test_sample.shape[1], test_sample.shape[2]))

        predicts =  model.predict(test_sample)
        mcs_tmp=predicts[:,1]/(predicts[:,1]+predicts[:,0])
        ef_tb.append(mcs_tmp*tb_set)
        c_pre.append(mcs_tmp)
        c_cmcs.append(c_mcs)
        a_s.append(a_test[sample_index])
    path_data = os.path.join(path, 'data/result_ack.pkl')
    with open(path_data, 'wb') as df:
        pickle.dump((c_pre,ef_tb, a_s,c_cmcs), df)
    print(len(c_pre))






