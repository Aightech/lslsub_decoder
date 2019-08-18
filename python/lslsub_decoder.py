import pandas as pd
import numpy as np
import get_data as tool
import matplotlib.pyplot as plt

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.externals import joblib

import scipy as sp
from scipy import signal


print('# LDA')
path='../16_07_19_Fio/raw_data/'
date='1607'
data=np.array([[]])

id = '222'
data = tool.get_data(date, id, path)
print(data.shape)

t = data[:,0]
y = data[:,1]
X = data[:,12:]

print("Loading LDA")
LDA=joblib.load('lda1.fil')

fe = 1300
nyq = fe/2

order_fil = 2
high_fil, low_fil = 450/nyq, 20/nyq
b_fil, a_fil = sp.signal.butter(order_fil, [low_fil, high_fil], btype='bandpass',output='ba')
x_fil = np.zeros((max(len(a_fil),len(b_fil)),256))
y_fil = np.zeros((max(len(a_fil),len(b_fil)),256))

order_env = 2
low_env = 3/nyq
b_env, a_env = sp.signal.butter(order_env, low_env, btype='low',output='ba')
x_env = np.zeros((max(len(a_env),len(b_env)),256))
y_env = np.zeros((max(len(a_env),len(b_env)),256))

def filterRT(b,a,x,y,ind):
    y[ind]=0
    for i in range(len(a)):
        y[ind] += b[i]*x[ind-i] - a[i]*y[ind-i]
    return x,y

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

    if(i%1000==0):
        print(i)
    y_pred.append(LDA.predict(sample))

plt.plot(y_pred)
plt.show()
print('Success rate : ' + str(tool.success_rate(y,y_pred)))

    


