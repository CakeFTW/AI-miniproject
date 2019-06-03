import pandas as pd
import numpy as np
import dataprocess as dp
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split

#variables for the LSTM
input_step = 5
input_features = 50
#prepare data
data_pandas = pd.read_csv('dataframe_data.csv') 
data = data_pandas.values
data = dp.remove_confidence_intervals(data)
#data = dp.relativeToFirstIndex(data)
#get labels
labels = pd.read_csv('dataframe_labels.csv').values
labels = dp.remove_confidence_intervals(labels)



#create train and test set

x_train,x_test,y_train,y_test = train_test_split(data,labels, test_size=0.2,random_state=4)

#now stack the data sets

traindata = np.zeros((x_train.shape[0], input_step, input_features))
for i in range(x_train.shape[0]):
    for j in range(input_step-1,-1,-1):
        traindata[i,j] = x_train[i, input_features*j : input_features*j + input_features]

testdata = np.zeros((x_test.shape[0], input_step, input_features))
for i in range(x_test.shape[0]):
    for j in range(input_step-1,-1,-1):
        traindata[i,j] = x_test[i, input_features*j : input_features*j + input_features]


print(traindata.shape)


#create the model
model = Sequential()
model.add(LSTM(125, activation = 'elu',  return_sequences=True, input_shape = (input_step, input_features)))
model.add(LSTM(125,activation = 'elu'))
model.add(Dense(input_features))
model.compile(optimizer="adam", loss= 'mse')

#fit the model 
model.fit(traindata,y_train, epochs= 5)

scores = model.evaluate(testdata, y_test)
print(scores)
pass

model.save('lstm_simple.h5')

