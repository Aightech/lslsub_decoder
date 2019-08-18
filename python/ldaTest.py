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

for id in ['222']:
    data_ = tool.get_processed_data(date, id, path)
    if data.size==0:
        data = data_
    else:
        data = np.concatenate((data,data_), axis=0)


t = data[:,0]
y = data[:,1]
X = data[:,12:]


print("Loading LDA")
LDA=joblib.load('lda1.fil')


print("Prediction")
y_pred = LDA.predict(X)
y_pred_fil = tool.average_label(y_pred,100)


print('Success rate : ' + str(tool.success_rate(y,y_pred)))
print('Success rate filtered: '  + str(tool.success_rate(y,y_pred_fil)))

fig = plt.figure(1)
plt.plot(t, y)
plt.plot(t, y_pred_fil)


fig = plt.figure(2)
tool.plot_coef(fig,LDA)


mat = tool.conf_mat(y,y_pred_fil,log=True)
    
    
fig = plt.figure(3)
plt.imshow(mat)
plt.colorbar()

fig = plt.figure(6)

mat = tool.conf_mat(y,y_pred_fil,log=False)
success_rate = [ mat[i,i]*100/np.sum(mat[i,:]) for i in range(len(mat))]
print(success_rate)
plt.bar( np.arange(len(success_rate)) , success_rate, align='center', alpha=0.5)


name=['None - 0',
      'Thumb down - 1',
      'Index down - 2',
      'Middle down - 3',
      'Ring down - 4',
      'Pinky down - 5',
      'Grasp - 6',
      'Horn 3 - 7',
      'Horn 2 - 8',
      'Phone - 9',
      'Pinch - 10',
      'Number 3 - 11',
      'Gun - 12',
      'Victory - 13',
      'Thumb-index - 14',
      'Thumb-middle - 15',
      'Thumb-ring - 16',
      'Thumb-pinky - 17',
      'Thumb up - 18',
      'index_up - 19'
]





plt.xticks(np.arange(len(success_rate)), name, rotation=20)

plt.show()


