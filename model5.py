import pandas as pd
import numpy as np
import dataprocess as dp
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.optimizers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sys import argv

input_step = 5
input_features = 63
out_shape = 50


def test(k_fold = False):

    if k_fold:
        print("performing kfold validation")
    else:
        print("test with split set")

    #variables for the LSTM



    if k_fold:
        data_pandas = dp.remove_confidence_intervals(pd.read_csv('dataframe_data.csv').values)
        #stack the values 
        data = np.zeros((data_pandas.shape[0], input_step, input_features))
        for i in range(data_pandas.shape[0]):
            for j in range(input_step-1,-1,-1):
                data[i,j] = data_pandas[i, input_features*j : input_features*j + input_features]
        #load labels
        labels = pd.read_csv('dataframe_labels.csv').values
        labels = dp.remove_confidence_intervals(labels)

        #generate folds
        indicies = np.arange(data_pandas.shape[0])
        np.random.seed(42)
        np.random.shuffle(indicies)
        fold_indicies = np.array_split(indicies, 10)


    else:
        #prepare data
        x_train = dp.remove_confidence_intervals(pd.read_csv('split_train_data.csv').values)
        x_test = dp.remove_confidence_intervals(pd.read_csv('split_test_data.csv').values)
        # data = dp.calcAnglesBody(data)
        # x_test = dp.relativeToFirstIndex(x_test, nrOfFeatures=50)
        # x_train = dp.relativeToFirstIndex(x_train,nrOfFeatures= 50)
        x_test_angle = dp.calcAnglesBody(x_test)
        x_train_angle = dp.calcAnglesBody(x_train)

        
        pd.DataFrame(x_test).to_csv("noci_data.csv")

        #get labels
        y_train = dp.remove_confidence_intervals(pd.read_csv('split_train_labels.csv').values)
        y_test = dp.remove_confidence_intervals(pd.read_csv('split_test_labels.csv').values)
        # labels = dp.calcAnglesBody(labels, number_of_frames=1 )
        pd.DataFrame(y_test).to_csv("noci_labels.csv")

        print(x_test)

        #calculate mse
        mse = ((x_test[:,12]-x_test[:,62])**2).mean()
        print("mse ===== ", mse)

        traindata = np.zeros((x_train.shape[0], input_step, 50))
        for i in range(x_train.shape[0]):
            for j in range(input_step-1,-1,-1):
                traindata[i,j] = x_train[i, 50*j : 50*j + 50]
        traindata_angles = np.zeros((x_train.shape[0], input_step, 13))
        for i in range(x_train.shape[0]):
            for j in range(input_step-1,-1,-1):
                traindata_angles[i,j] = x_train[i, 13*j : 13*j + 13]

        testdata = np.zeros((x_test.shape[0], input_step, 50))
        for i in range(x_test.shape[0]):
            for j in range(input_step-1,-1,-1):
                testdata[i,j] = x_test[i, 50*j : 50*j + 50]

        testdata_angles = np.zeros((x_test.shape[0], input_step, 13))
        for i in range(x_test.shape[0]):
            for j in range(input_step-1,-1,-1):
                testdata_angles[i,j] = x_test[i,13*j : 13*j + 13]

        traindata =np.nan_to_num( np.append(traindata,traindata_angles,2))
        testdata = np.nan_to_num(np.append(testdata, testdata_angles,2))

        print(testdata.shape)
        print(testdata_angles.shape)
        



    #create the model

    if k_fold:
        k_fold_scores = np.arange(10)
        for fold in range(10):

            model = createModel()

            trainmask = np.ones(data.shape[0], dtype= bool)
            trainmask[fold_indicies[fold]] = False

            testmask = np.zeros(data.shape[0], dtype= bool)
            testmask[fold_indicies[fold]] = True

            model.fit(data[trainmask], labels[trainmask], validation_data=(data[testmask], labels[testmask]), epochs = 1)
            scores = model.evaluate(data[testmask], labels[testmask])
            k_fold_scores[fold] = scores
        print(k_fold_scores)
        return np.average(k_fold_scores)

    else:
        model = createModel()
        model.fit(traindata,y_train, validation_data=(testdata,y_test), epochs= 1)
        scores = model.evaluate(testdata, y_test)
        print(scores)

        outs = model.predict(x_test)
        print(outs.shape)
    # pass

    return scores

def createModel():
        #create the model
    model = Sequential()
    # model.add(Dense(25,input_shape = (input_step, input_features)))
    # model.add(Dense(25))
    model.add(LSTM(25, activation = 'elu', return_sequences=True, input_shape = (input_step, input_features)))
    model.add(LSTM(25, activation = 'elu'))
    model.add(Dense(out_shape))
    model.compile(optimizer='adam', loss= 'mse')
    # model.summary()
    return model

test(False)