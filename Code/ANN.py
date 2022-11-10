# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 03:56:04 2020

@author: sas11
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf # Imports tensorflow
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras import optimizers
import seaborn as sns
from tensorflow.keras.utils import plot_model

#data= pd.read_csv("dataset2.csv")
column_names = ['temp','humi','rpm']
dataset = pd.read_csv("dataset.csv", names=column_names,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)

print(dataset)
sns.pairplot(dataset[["temp", "humi","rpm"]], diag_kind="kde")

data = np.asarray(dataset)
print(data)
x=[]
y=[]
for i in tqdm(range(data.shape[0])):
    x.append([data[i][0],data[i][1]])
    y.append(data[i][2])
x=np.array(x)
y=np.array(y)
print(x.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
model = Sequential()
model.add(Dense(16, activation='relu', input_dim= 2,kernel_initializer='he_uniform'))
model.add(BatchNormalization(axis=-1))
#model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
#model.add(BatchNormalization(axis=-1))
#model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
#model.add(BatchNormalization(axis=-1))
#model.add(Dropout(0.5))

model.add(Dense(1, activation='linear', kernel_initializer='normal' ))

model.summary()
plot_model(model, to_file='ANN.png',show_shapes= True , show_layer_names=True)

model.compile(loss='msle',
                optimizer='adam',
                metrics=['mae', 'msle','acc'])
history = model.fit(X_train, y_train,
          batch_size=500,
          epochs=100,
          validation_data=(X_test, y_test), 
          verbose=1)

#model.save('ann2.h5')
prd =model.predict(x)
a= np.asarray(prd)
print(prd)
pd.DataFrame(a).to_csv("predAnn.csv")
pd.DataFrame.from_dict(history.history).to_csv('ann2.csv',index=False)
