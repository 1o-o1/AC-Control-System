import numpy as np
import math
from network.anfisnetwork import Anfis
from network.controller import Controller
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import seaborn as sns


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

model = Anfis(num_rules=5, input_dims=2, output_classes=1)
data = {'X_train': X_train,
        'y_train': y_train,
        'X_val': None,
        'y_val': None}

controller = Controller(model, data, print_every=100, num_epochs=1, batch_size=100, update_rule='sgd',
                        optim_config={
                            'learning_rate': 0.01
                        })

controller.train()
# mask = np.random.choice(X_train.shape[0], 6)
# X_test = X_train[mask]
# y_test = y_train[mask]

prediction = controller.predict(X_test)
print(prediction)
print(y_test)
