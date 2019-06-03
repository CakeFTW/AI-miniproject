from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import os
import sys
import pandas
import json
import numpy as np
import matplotlib.pyplot as plt
from math import floor
import dataprocess as dp
import pandas as pd


data_set = pd.read_csv("dataframe_data.csv")

data_set_np = data_set.values
data_set_np = dp.circle_scale(data_set_np)

data_labels = pd.read_csv("dataframe_labels.csv")
data_labels['labels'] = data_labels['0'].astype(str)
print(len(data_set_np))
print(len(data_set_np[0]))
data = data_set_np.reshape(1, data_set_np.shape[0], data_set_np.shape[1])

x_train,x_test,y_train,y_test = train_test_split(data,data_labels,test_size=0.2,random_state=4)
model=Sequential()
model.add(LSTM((1),batch_input_shape=(None,5,1),return_sequences=True))
model.add(LSTM((1),return_sequences=False))
print(target)
model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
history=model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))
results=model.predict(x_test)
print(len(results))
print(len(y_test))
plt.scatter(range(15),results,c='r')
plt.scatter(range(15),y_test,c='g')
plt.show()
plt.plot(history.history['loss'])
plt.show()

#