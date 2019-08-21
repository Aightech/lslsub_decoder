import pandas as pd
import numpy as np
import get_data as tool
import matplotlib.pyplot as plt

import time
from pylsl import StreamInfo, StreamOutlet

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib

import scipy as sp
from scipy import signal


lstm=True
path='../16_07_19_Fio/raw_data/'
date='1607'
data=np.array([[]])

id = '112'
data = tool.get_data(date, id, path)
print(data.shape)


t = data[:,0]
y = data[:,1]
X = data[:,12:]

x_scaler = joblib.load('Xscaler.sca')

if(lstm==True):
    print("Loading LSTM")
    model = tf.keras.models.load_model('lstm06.h5')
    y_scaler = joblib.load('Yscaler.sca')
    step = 5
    step_arr = np.zeros((step,256))
else:
    print("Loading LDA")
    model=joblib.load('lda1.fil')

fe = 1300
nyq = fe/2

order_fil = 4
high_fil, low_fil = 450/nyq, 20/nyq
b_fil, a_fil = sp.signal.butter(order_fil, [low_fil, high_fil], btype='bandpass',output='ba')
x_fil = np.zeros((max(len(a_fil),len(b_fil)),256))
y_fil = np.zeros((max(len(a_fil),len(b_fil)),256))

order_env = 4
low_env = 3/nyq
b_env, a_env = sp.signal.butter(order_env, low_env, btype='low',output='ba')
x_env = np.zeros((max(len(a_env),len(b_env)),256))
y_env = np.zeros((max(len(a_env),len(b_env)),256))

def filterRT(b,a,x,y,ind):
    y[ind]=0
    for i in range(len(a)):
        y[ind] += b[i]*x[ind-i] - a[i]*y[ind-i]
    return x,y

info = StreamInfo('Left_Hand_Command', 'hand_position', 15, 0, 'float32')
outlet = StreamOutlet(info)

count = 0
mean = 0
y_pred = []
for i in range(len(X)):
    sample = X[i]
        
    # 1. Mean rectified 
    mean = (mean*count + sample)
    count += 1
    mean /= count
    sample -= mean

    # 2. Filter 
    x_fil[i%len(a_fil)]= sample
    x_fil, y_fil = filterRT(b_fil, a_fil, x_fil, y_fil,i%len(a_fil))

    # 3. Rectify
    sample = abs(y_fil[i%len(a_fil)])

    # 4. Envelope
    x_env[i%len(a_env)]= sample
    x_env, y_env = filterRT(b_env, a_env, x_env, y_env,i%len(a_env))
    sample = np.array(abs(y_env[i%len(a_env)]), ndmin=2)
    
    if(lstm==True):
        step_arr[i%step] = x_scaler.transform(sample)
        sample = np.concatenate( (step_arr[(i+1)%step:],step_arr[:(i+1)%step]) )
        sample = np.reshape(sample, (1,step,256))

    y_pred.append(model.predict(sample))

    if(lstm):
        mysample = y_scaler.inverse_transform(y_pred[-1])
        mysample = [mysample[0,j]/100 for j in [0,0,1, 2,2,3, 4,4,5, 6,6,7, 8,8,9]]
        outlet.push_sample(mysample)
    
    if(i%1000==0):
        print(i)



if(lstm==True):
    fig = plt.figure()
    y_pred=np.reshape(np.array(y_pred),(len(y_pred),10))
    print(y_pred.shape)
    for i in range(len(y_pred[0,:])):
        ax = fig.add_subplot(5,2,i+1)
        ax.plot(t,y_pred[:,i])
        ax.plot(t,y[:,i])
    plt.show()
else:
    plt.plot(y_pred)
    plt.plot(y)
    plt.show()
    print('Success rate : ' + str(tool.success_rate(y,y_pred)))

    


