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
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
#data= pd.read_csv("dataset2.csv")
column_names = ['temp','humi','rpm']
dataset = pd.read_csv("dataset.csv", names=column_names,
                      na_values = "?", comment='\t',
                      sep=",", skipinitialspace=True)

print(dataset)
#sns.pairplot(dataset[["temp", "humi","rpm"]], diag_kind="kde")

data = np.asarray(dataset)
#print(data)
x=[]
y=[]
for i in tqdm(range(data.shape[0])):
    x.append([data[i][0],data[i][1]])
    y.append(data[i][2])
x=np.array(x)
y=np.array(y)
print(x.shape)
print(y.shape)

model = keras.models.load_model('ann2.h5')
plot_model(model, to_file='ANN.png',show_shapes= True , show_layer_names=True)
model.summary()
pr=model.predict(x)
a= np.asarray(pr)
a= np.reshape(a,100000)
for i in range(100000):
    if a[i]<0:
        a[i]=0
    elif a[i]>7200:
        a[i]=7200
        
#pd.DataFrame(a).to_csv("predAnn2.csv")
print(a)
plt.plot(a[:100])
plt.plot(y[:100])
plt.title('pred vs actual')
#plt.ylim(0,0.5)
#plt.xlim(-4,78)
plt.ylabel('Rpm')
plt.xlabel('data')
plt.legend(['prediction', 'True class'], loc='upper left')
plt.show()
