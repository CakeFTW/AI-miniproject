import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data_pd = pd.read_csv('k_fold_scores.csv')

# print(data_pd)

# model = data_pd['model'].values
# zero = data_pd['pred'].values

# print(model)

# fig1, ax1 = plt.subplots()
# ax1.set_title('Model 21 vs Zero velocity predition')
# ax1.boxplot([model,zero], notch = True)
# ax1.set_ylabel("MSE")
# ax1.set_xticklabels(('model21', 'zero velocity predictor'))

split = pd.read_csv('hist_loss.csv')
kfold = pd.read_csv('nosplit_loss.csv')
print(split)

plt.plot(split['loss'])
plt.plot(split['val_loss'])
plt.plot(kfold['loss'])
plt.plot(kfold['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['per-video split: train', 'per-video split: test', 'per-json split: train', 'per-json: test'], loc='upper right')
plt.show()