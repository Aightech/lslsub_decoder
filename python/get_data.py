import pandas as pd
import numpy as np
import os

import scipy as sp
from scipy import signal
import sklearn.model_selection as sk
import matplotlib.pyplot as plt

def get_processed_data(date, id, path):
     if os.path.isfile(path+'all_id'+date+'_'+ id +'_processed.csv')== True:
        print("Importing : " + path+'all_id'+date+'_'+ id +'_processed.csv')
        return  pd.read_csv(path+'all_id'+date+'_'+ id +'_processed.csv',  header=None, sep=',').to_numpy()
     else:
         data = get_data(date, id, path)
         
         print("Creating : " + path+'all_id'+date+'_'+ id +'_processed.csv')
         t=data[:,0:1]
         l=data[:,1:2]
         y=data[:,2:12]
         X=data[:,12:]

         # ch=0
         # ts=19000
         # te=20500
         # fig = plt.figure(1)
         # fig.add_subplot(5,1,1)
         # plt.plot(t[ts:te],X[ts:te,ch])

         ## 1. Mean rectified
         X = X - np.mean(X, axis=0)
         print('> Mean Corrected.')

         # fig.add_subplot(5,1,2)
         # plt.plot(t[ts:te],X[ts:te,ch])
         
         ## 2. Filtering 
         # Create bandpass filter for EMG
         fe = len(t)/(t[-1]-t[0])
         nyq = fe/2
         high = 450/nyq
         low = 20/nyq
         order = 4
         sos = sp.signal.butter(order, [low,high], btype='bandpass',output='sos')
         X = sp.signal.sosfilt(sos, X, axis=0)
         print('> Filtered.')

         # fig.add_subplot(5,1,3)
         # plt.plot(t[ts:te],X[ts:te,ch])

         # 3 Rectified
         X = abs(X)
         print('> Rectified.')

         # fig.add_subplot(5,1,4)
         # plt.plot(t[ts:te],X[ts:te,ch])
         
         #enveloppe
         low = 3/(nyq)
         sos = sp.signal.butter(order, low, btype='low',output='sos')
         X = sp.signal.sosfilt(sos, X, axis=0)
         print('> Envelope.')

         # fig.add_subplot(5,1,5)
         # plt.plot(t[ts:te],X[ts:te,ch])
         # plt.show()

         X = np.concatenate((t,l,y,X), axis=1)
         print(X.shape)
         df = pd.DataFrame(X)
         df.to_csv(path+'all_id'+date+'_'+ id +'_processed.csv',header=False,index=False)
         return X
         

def get_data(date, id, path):
    if os.path.isfile(path+'all_id'+date+'_'+ id +'.csv')== False:
        print("Creating : " + path+'all_id'+date+'_'+ id +'.csv')
        ## Get the glove data ##
        df = pd.read_csv(path+'glv_id'+date+'_'+ id +'.csv',  header=None, sep=',')
        df = df.sort_values(0)#ensure the row are well sorted
        T_glv = df.iloc[:, 1].copy().to_numpy()#time column
        X_glv = df.iloc[:, 2:17].copy().to_numpy()#data
        X_glv = np.delete(X_glv, np.s_[1::3], 1)#remove repeated column
        T_glv = np.array(T_glv,ndmin=2).T#reshape to concatenate
        
        ## Get the OTB data ##
        df = pd.read_csv(path+'otb408_id'+date+'_'+id+'.csv',  header=None, sep=',')
        df = df.sort_values(0)#ensure the rows are welll sorted
        T_otb = df.iloc[:,1].copy().to_numpy()#time column
        X_otb = df.iloc[:,2+8*16-11:-8-16].copy().to_numpy()# keep only the mulinput data and keep space for glv data
        
        # store glv data at the right time
        j=0
        size = len(T_glv)
        for i in range(len(T_otb)):
            if(j+1< size and T_otb[i]>T_glv[j]):
                j+=1
            X_otb[i,1:11]=X_glv[j,:]*100
        
        y=X_otb[:,1:11]
        
        ## Labelize and find the segmentation ##
        avg_step = 400
        clip_min = 4000
        clip_max = 14000
        label = [np.max(y[i,:])+y[i,8]+2*y[i,2]
                 for i in range(len(y))] # Create a signal to ease the labelization
        label = [np.sum(label[i:min(i+avg_step, len(label))])/len(label[i:min(i+avg_step, len(label))])
                 for i in range(len(label))]# Average to avoid close peak
        label = [(max(min(x, clip_max), clip_min)-clip_min)/(clip_max-clip_min)
                 for x in label] # Clip in the interesting range
        
        label_temp = label
        lab=0
        offset=-0.3
        while(lab!=19):
            mean = np.mean(label_temp)+offset
            label = [1 if x > mean else 0 for x in label_temp] ## binarize
            # Segmentation
            label[0] = 1 if label[0] == 1 else 0
            lab = label[0]
            for i in range(1,len(label)):
                if label[i]==1:
                    if label[i-1] == 0:
                        lab+=1
                    label[i]=lab
                else:
                    label[i]=0
            print('Found ' + str(lab) + ' segments.')
            offset+=0.05
        
        plt.plot(T_otb, label_temp)
        plt.hlines(mean,T_otb[0],T_otb[-1])
        plt.show()
        
        ## Store and save
        X_otb[:,0]=label
        T_otb=np.array(T_otb,ndmin=2).T
        X=np.concatenate((T_otb, X_otb), axis=1)
        df = pd.DataFrame(X)
        df.to_csv(path+'all_id'+date+'_'+ id +'.csv',header=False,index=False)
        return X
    else:
        print("Importing : " + path+'all_id'+date+'_'+ id +'.csv')
        return  pd.read_csv(path+'all_id'+date+'_'+ id +'.csv',  header=None, sep=',').to_numpy()       
    
def train_test_split(X,y,t,test_size, order=True):
    if(order==False):
        y=np.concatenate((np.array(t,ndmin=2).T, np.array(y,ndmin=2).T), axis=1)
        X_train, X_test, y_train, y_test = sk.train_test_split(X,y,test_size=test_size)

        y_train, y_test = y_train[:,1:], y_test[:,1:]
        t_train, t_test = y_train[:,0], y_test[:,0]
        return X_train, X_test, y_train, y_test, t_train, t_test
    else:
        y=np.concatenate((np.array(t,ndmin=2).T, np.array(y,ndmin=2).T), axis=1)
        print(X.shape,y.shape)
        X_train, X_test, y_train, y_test = sk.train_test_split(X,y,test_size=test_size) 
        
        size = len(y_train[0,:])
        
        arr = np.concatenate((y_train,X_train),axis=1)
        arr = arr[arr[:,0].argsort(),:]
        t_train, y_train, X_train = arr[:,0], arr[:,1:size], arr[:,size:]

        arr = np.concatenate((y_test,X_test),axis=1)
        arr = arr[arr[:,0].argsort(),:]
        t_test, y_test, X_test = arr[:,0], arr[:,1:size], arr[:,size:]

        if(size==2):
            y_train, y_test = np.ravel(y_train), np.ravel(y_test)

        return X_train, X_test, y_train, y_test, t_train, t_test

def average_label(l,avg):
    s=len(l)
    return [np.unique(l[i:min(i+avg,s)],
                      return_counts=True)[0][np.argmax(np.unique(l[i:min(i+avg,s)],
                                                                 return_counts=True)[1])]
            for i in range(s)]

def success_rate(y,y_pred):
    return np.sum([y[i]==y_pred[i] for i in range(len(y))])/len(y)

    
def plot_coef(fig,LDA):
    for i in range(len(LDA.coef_)):
      ax = fig.add_subplot(5,4,i+1)
      plt.imshow(np.concatenate( [np.reshape(LDA.coef_[i,j*64:(j+1)*64],(8,8)) for j in range(4)], axis=1 ))

def conf_mat(y,yp,log=False):
    mat = np.zeros((len(np.unique(y)),len(np.unique(y))),dtype=np.float64)
    for i in range(len(y)):
        mat[int(y[i]),int(yp[i])]+=1
    if(log):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i,j] = 0 if mat[i,j]==0 else np.log(mat[i,j])
    return mat

