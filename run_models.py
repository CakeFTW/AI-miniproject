# import lstm_simple
import test_lib0 as model
from subprocess import call
from sys import argv
import csv
import time
import pandas as pd
file_name="models_analytics.csv"
csv_input = pd.read_csv(file_name)

description = """
Layers:
(LSTM'25)
(LSTM'25)
(LSTM'25)
OUT-(Dense'50)
Optimizer: Adam
Data Layout: Screen space, 50 values in, 50 values out
Learning Rate: Default
Comment: First model
"""
# with open(file_name, 'rt') as csvfile:
#     reader=csv.reader(csvfile)
#     data=list(reader)
#     print(data)
#     row0=next(reader)
#     row0.append(description)
# with open(file_name, 'wt') as csvfile:
#     filewriter = csv.writer(csvfile)
#     old_header=data[0][:]
#     print(old_header)
#     new_header=[old_header, description.replace(","," ")]
#     filewriter.writerow(new_header)
tmp_arr=[]
for test_type in range(1):
    if(test_type==96758):
        kfold=True
    else:
        kfold=False
    for x in range(20):
        print("///////////////ITERATION NUMBER: ",str(x+1)," OUT OF 20///////////////")
        tmp_str=model.test(kfold)
        tmp_arr.append(tmp_str)
print(tmp_arr)
csv_input[description.replace(","," ")]=tmp_arr
csv_input.to_csv(file_name, index=False)