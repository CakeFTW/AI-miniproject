import pandas as pd
import numpy as np
import dataprocess as dp
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sys import argv

k_fold = False
if argv[1] == '1':
        print("performing kfold validation")
        k_fold = True
else:
        print("test with split set")

#variables for the LSTM
input_step = 5
input_features = 50

#prepare data
data_pandas = pd.read_csv('dataframe_data.csv') 
data = data_pandas.values
data = dp.remove_confidence_intervals(data)
# data = dp.calcAnglesBody(data)
# data = dp.relativeToFirstIndex(data, nrOfFeatures=50)
#get labels
labels = pd.read_csv('dataframe_labels.csv').values
labels = dp.remove_confidence_intervals(labels)
# labels = dp.calcAnglesBody(labels, number_of_frames=1 )



#create train and test set

x_train,x_test,y_train,y_test = train_test_split(data,labels, test_size=0.5,random_state=4)

#now stack the data sets

traindata = np.zeros((x_train.shape[0], input_step, input_features))
for i in range(x_train.shape[0]):
    for j in range(input_step-1,-1,-1):
        traindata[i,j] = x_train[i, input_features*j : input_features*j + input_features]

testdata = np.zeros((x_test.shape[0], input_step, input_features))
for i in range(x_test.shape[0]):
    for j in range(input_step-1,-1,-1):
        testdata[i,j] = x_test[i, input_features*j : input_features*j + input_features]



#create the model
model = Sequential()
# model.add(Dense(100, ))
# model.add(Dense(100))
model.add(LSTM(100, return_sequences=True,input_shape = (input_step, input_features)))
model.add(LSTM(50))
model.add(Dense(labels.shape[1]))
model.compile(optimizer="adam", loss= 'mse')
# model.summary()

#fit the model 
model.fit(traindata,y_train, validation_data=(testdata,y_test), epochs= 20)

scores = model.evaluate(testdata, y_test)

print(scores)
# pass

exit(scores)
