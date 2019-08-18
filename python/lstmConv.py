import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib


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
        a=np.reshape(a,(a.shape[0],1,8,8*4,1))
        cols.append(a)
    agg = np.concatenate(cols, axis=1)
    agg=np.delete(agg,np.s_[0:step-1],0)
    return agg

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, X_data, y_data , batch_size=32, dim=(8,32), dim_out=(10,),n_channels=1, time_step=2, shuffle=True, size=-1):
        'Initialization'
        self.dim = dim
        self.dim_out = dim_out
        self.batch_size = batch_size
        self.X_data = X_data
        self.y_data = y_data
        self.n_channels = n_channels
        self.time_step = time_step
        self.shuffle = shuffle
        self.size = size if size!=-1 else len(self.y_data)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.size / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size-self.time_step:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.y_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.time_step, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim_out))

        # Generate data
        for i in range(len(indexes)-self.time_step):
            # Store sample
            X[i,] = np.concatenate( [  np.reshape(self.X_data[indexes[i+self.time_step-j],:]
                                                  ,(1,*self.dim,self.n_channels))
                                       for j in range(self.time_step,0,-1)                     ], axis=0)

            # Store class
            y[i,] = self.y_data[i,:]

        return X, y

def fit_lstm(X_data, y_data, step=2, epochs=5, validation_split=0.2,batch_size=32):
    #normalize the dataset
    nb_features = len(X_data[0,:])

    model = keras.Sequential([
        layers.ConvLSTM2D(64, (3, 3), strides=(1, 1),  input_shape=(step,8,4*8,1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(10)
    ])
    
    model.compile(loss='mean_absolute_error',
                  optimizer='Adadelta',
                  metrics=['mean_absolute_error','mean_absolute_error'])
    
    training_generator = DataGenerator(X_data, y_data , batch_size=batch_size, dim=(8,32), dim_out=(10,), n_channels=1, time_step=step, shuffle=True,size=10000)
    validation_generator = DataGenerator(X_data, y_data , batch_size=batch_size, dim=(8,32), dim_out=(10,),n_channels=1, time_step=step, shuffle=True, size= 10000)
    
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        workers=6,
                        epochs=epochs)
    #model.fit(X_data, y_data, epochs=epochs, validation_split = validation_split)

    return model


path='../16_07_19_Fio/raw_data/'
date='1607'
data=np.array([[]])
id ='111'

for id in['111','112','121','122']:#,'211','212','221','222']:
    data_ = g.get_data(date, id, path)
    print(id)
    if data.size==0:
        data = data_
    else:
        data = np.concatenate((data,data_), axis=0)



scaler = MinMaxScaler(feature_range=(0, 1))
joblib.dump(scaler,'scaler.sca')
data = scaler.fit_transform(data.astype('float32'))


t = data[:,0]
y = data[:,1:11] 
X = data[:,11:] 

step=10
batch_size=100
epochs=10
model = fit_lstm(X, y, step=step, epochs=epochs, batch_size=batch_size)
model.save('lstm16_07_0828.h5')
