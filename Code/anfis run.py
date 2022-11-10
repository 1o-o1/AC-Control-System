import anfis
import membership.mfDerivs
import membership.membershipfunction
from skfuzzy import gaussmf, gbellmf, sigmf
import numpy as np
import itertools
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import tensorflow as tf # Imports tensorflow
import tensorflow_addons as tfa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras import optimizers
import seaborn as sns
import copy


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
for i in tqdm(range(10000)):
    x.append([data[i][0],data[i][1]])
    y.append(data[i][2])


X=np.array(x)
Y=np.array(y)

mf = [[['gaussmf',{'mean':0.,'sigma':1.}],['gaussmf',{'mean':-1.,'sigma':2.}],['gaussmf',{'mean':-4.,'sigma':10.}],['gaussmf',{'mean':-7.,'sigma':7.}]],
            [['gaussmf',{'mean':1.,'sigma':2.}],['gaussmf',{'mean':2.,'sigma':3.}],['gaussmf',{'mean':-2.,'sigma':10.}],['gaussmf',{'mean':-10.5,'sigma':5.}]]]


mfc = membership.membershipfunction.MemFuncs(mf)
anf = anfis.ANFIS(X, Y, mfc)
anf.trainHybridJangOffLine(epochs=50)
#print(round(anf.consequents[-1][0],6))
#print(round(anf.consequents[-2][0],6))
#print(round(anf.fittedValues[9][0],6))
#if round(anf.consequents[-1][0],6) == -5.275538 and round(anf.consequents[-2][0],6) == -1.990703 and round(anf.fittedValues[9][0],6) == 0.002249:
#	print('test is good')
anf.plotErrors()

#anf.plotResults()
pred = anf.fittedValues
a= np.asarray(pred)
print(pred)
anf.plotMF(X,1)
pd.DataFrame(a).to_csv("predAnfisfinal.csv")
