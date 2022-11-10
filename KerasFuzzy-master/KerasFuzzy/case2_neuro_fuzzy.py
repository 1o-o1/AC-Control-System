
from FuzzyLayer import FuzzyLayer
from DefuzzyLayer import DefuzzyLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
# Generate dummy data
import numpy as np

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
model.add(Dense(32,activation='relu',input_dim=2,kernel_initializer='he_uniform'))
model.add(FuzzyLayer(16))
model.add(Dense(64, activation='relu'))
model.add(DefuzzyLayer(32))
model.add(Dense(1,activation='linear',kernel_initializer='normal'))
model.compile(loss='msle',
              optimizer='adam',
              metrics=['mae', 'acc','msle'])

model.fit(X_train, y_train,
          epochs=100,
          verbose=1,
          batch_size=1000,validation_data=(X_test, y_test))

prd =model.predict(x)
a= np.asarray(prd)
print(prd)
pd.DataFrame(a).to_csv("predAnfis6.csv")


print('Done')