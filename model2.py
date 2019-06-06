import pandas as pd
import numpy as np
import dataprocess as dp
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.optimizers import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sys import argv

input_step = 5
input_features = 50
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
        x_train = dp.remove_confidence_intervals((pd.read_csv('split_train_data.csv').append(pd.read_csv('split_test_data.csv'))).values)
        x_test = dp.remove_confidence_intervals(pd.read_csv('split_test_data.csv').values)
        x_train
        # data = dp.calcAnglesBody(data)
        # x_test = dp.relativeToFirstIndex(x_test, nrOfFeatures=input_features)
        # x_train = dp.relativeToFirstIndex(x_train, nrOfFeatures=input_features)

        #get labels
        y_train = dp.remove_confidence_intervals((pd.read_csv('split_train_labels.csv').append(pd.read_csv('split_test_labels.csv'))).values)
        y_test = dp.remove_confidence_intervals(pd.read_csv('split_test_labels.csv').values)
        # labels = dp.calcAnglesBody(labels, number_of_frames=1 )

        mse = ((y_test-x_test[:,0:50])**2).mean()
        print("mse ===== ", mse)

        traindata = np.zeros((x_train.shape[0], input_step, input_features))
        for i in range(x_train.shape[0]):
            for j in range(input_step-1,-1,-1):
                traindata[i,j] = x_train[i, input_features*j : input_features*j + input_features]

        testdata = np.zeros((x_test.shape[0], input_step, input_features))
        for i in range(x_test.shape[0]):
            for j in range(input_step-1,-1,-1):
                testdata[i,j] = x_test[i, input_features*j : input_features*j + input_features]



    #create the model

    if k_fold:
        k_fold_scores = np.arange(10, dtype=np.float32)
        zero_pred_scores = np.arange(10, dtype=np.float32)
        print(k_fold_scores[4])
        for fold in range(10):

            model = createModel()



            trainmask = np.ones(data.shape[0], dtype= bool)
            trainmask[fold_indicies[fold]] = False

            testmask = np.zeros(data.shape[0], dtype= bool)
            testmask[fold_indicies[fold]] = True

            mseZero = ((labels[testmask]-data[testmask, 1 , :])**2).mean()

            history = model.fit(data[trainmask], labels[trainmask], validation_data=(data[testmask], labels[testmask]), epochs = 10)
            scores = model.evaluate(data[testmask], labels[testmask])
            

            y_predicted = model.predict(data[testmask])
            print("test time")
            mse = ((labels[testmask]-y_predicted)**2).mean()
            
            zero_pred_scores[fold] = mseZero
            print("mse ===== ", mse, "   ", mseZero)
            k_fold_scores[fold] = mse

            print(k_fold_scores)
        print(k_fold_scores)
        pd.DataFrame(k_fold_scores).to_csv("k_fold_scores.csv")
        pd.DataFrame(zero_pred_scores).to_csv("0_pred_scores.csv")
        return np.mean(k_fold_scores)

    else:
        model = createModel()
        history = model.fit(traindata,y_train, validation_split = 0.33, epochs=10 )
        y_predicted = model.predict(testdata)
        print(y_predicted.shape)
        pd.DataFrame(history.history).to_csv('nosplit_loss.csv')
        print("test time")
        mse = ((y_test-y_predicted)**2).mean()
        print("mse ===== ", mse)
        print(history.history.keys())
        

    # pass

    return scores

def createModel():
        #create the model
    model = Sequential()
    model.add(LSTM(50, activation = 'elu', input_shape = (input_step, input_features)))
    model.add(Dense(out_shape))
    optimi = SGD(0.1, momentum = 0.8,nesterov = True)
    model.compile(optimizer= optimi, loss= 'mse')
    model.summary()
    return model

print(test(False))

pass