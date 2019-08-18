import pandas as pd
import numpy as np
import get_data as tool
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.externals import joblib


import seaborn as sns

print('# LDA')
path='../16_07_19_Fio/raw_data/'
date='1607'
data=np.array([[]])

for id in ['111','112','121','211','212']:
    data_ = tool.get_processed_data(date, id, path)
    if data.size==0:
        data = data_
    else:
        data = np.concatenate((data,data_), axis=0)


t = data[:,0]
l = data[:,1]
y = data[:,2:12]
X = data[:,12:]

print("Splitting ...\n")
# 0. Load in the data and split the descriptive and the target feature
test_ratio=0.36
X_train, X_test, y_train, y_test, t_train, t_test = tool.train_test_split(X,l,t,test_size=test_ratio) 


#1. Instantiate the method and fit_transform the algotithm
n_comp=3
print("LDA ...\n")
LDA = LinearDiscriminantAnalysis(n_components=n_comp)
LDA.fit(X_train,y_train)
joblib.dump(LDA,'lda1.fil')
# print("Loading LDA")
# LDA=joblib.load('lda1.fil')


print("train prediction")
y_train_pred = LDA.predict(X_train)
y_train_pred_fil = tool.average_label(y_train_pred,100)


print("test prediction")
y_test_pred = LDA.predict(X_test)
y_test_pred_fil = tool.average_label(y_test_pred,100)

print('Success rate train: ' + str(tool.success_rate(y_train,y_train_pred)))
print('Success rate test: '  + str(tool.success_rate(y_test,y_test_pred)))
print('Success rate train filtered: ' + str(tool.success_rate(y_train,y_train_pred_fil)))
print('Success rate test filtered: '  + str(tool.success_rate(y_test,y_test_pred_fil)))

fig = plt.figure(1)
plt.plot(t_train, y_train)
plt.plot(t_train, y_train_pred_fil)

fig = plt.figure(2)
plt.plot(t_test, y_test)
plt.plot(t_test, y_test_pred_fil)

fig = plt.figure(3)
tool.plot_coef(fig,LDA)


mat_train = tool.conf_mat(y_train,y_train_pred_fil, log=True)
mat_test  = tool.conf_mat(y_test, y_test_pred_fil , log=True)
    
    
fig = plt.figure(4)
plt.imshow(mat_train)
fig = plt.figure(5)
plt.imshow(mat_train)



plt.show()


