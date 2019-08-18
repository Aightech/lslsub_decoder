import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


import pandas as pd
import numpy as np

import get_data as g
import matplotlib.pyplot as plt


def format(data,step=2):
    # reshape input to be 3D [samples, timesteps, features]
    cols=list()
    df= pd.DataFrame(data)
    for i in range(step-1, -1, -1):
        a=df.shift(i).to_numpy()
        a=np.reshape(a,(a.shape[0],1,a.shape[1]))
        cols.append(a)
    agg = np.concatenate(cols, axis=1)
    agg=np.delete(agg,np.s_[0:step-1],0)
    return agg

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

def fit_lstm(X_data, y_data, step=2, epochs=5, validation_split=0.2):
    #normalize the dataset
    nb_features = len(X_data[0,:])

    # reshape input to be 3D [samples, timesteps, features]
    X_data = format(X_data,step)
    y_data = y_data[step-1:,:]

    model = keras.Sequential([
        layers.LSTM(64,  input_shape=(step,nb_features)),
        layers.Dense(10)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    model.fit(X_data, y_data, epochs=epochs, validation_split = validation_split)

    return model


path='../16_07_19_Fio/raw_data/'
date='1607'
data=np.array([[]])
id ='111'

for id in ['112']:# ['112','122','132','142','152','212','222','232','242','252']:
    data_ = g.get_data(date, id, path)
    print(id)
    if data.size==0:
        data = data_
    else:
        data = np.concatenate((data,data_), axis=0)

scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data.astype('float32'))

train_ratio=0.17
train_size = int(len(data[:,0]) * train_ratio)
test_size = len(data[:,0]) - train_size
train, test = data[:train_size,:], data[train_size:,:]

t_train, t_test = train[:,0]   ,test[:,0]
y_train, y_test = train[:,1:11],test[:,1:11] 
X_train, X_test = train[:,11:] ,test[:,11:]

step=1
epochs=10
model = fit_lstm(X_train, y_train, step=step, epochs=epochs)
model.save('lstm06.h5')
#model = tf.keras.models.load_model('lstm06.h5')
X_test = format(X_test,step)
y_test = y_test[step-1:,:]
t_test = t_test[step-1:]
y_pred = model.predict(X_test)

fig = plt.figure()
for i in range(len(y_pred[0,:])):
    ax = fig.add_subplot(5,2,i+1)
    ax.plot(t_test,y_pred[:,i])
    ax.plot(t_test,y_test[:,i])

plt.show()

