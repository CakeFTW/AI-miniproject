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

try:
    open("keypoints.data")
except FileNotFoundError:
    print("keypoints.data not found")

output_file = open("keypoints.data",'r')

dataJson = json.load(output_file)
print(dataJson[6])
#uncomment this
Data = dataJson[:floor(0.8*len(dataJson))]
target= dataJson[len(dataJson) - floor(0.8*len(dataJson)):]
data=np.array(Data, dtype=float)
target=np.array(target, dtype=float)
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=4)
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