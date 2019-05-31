import os
from sys import argv
import json
from time import time
import pandas

start_time = time()
try:
    dirs =  os.listdir(argv[1])
except:
    print("no folder provided")

origin = os.getcwd()

paths_to_folders = []

data = []
labels = []

for path in dirs:
    os.chdir(origin + "/" + argv[1] + "/" + path)
    files = os.listdir()
    files.sort(key=os.path.getmtime)
    for x in range(len(files)):
        tmp_arr = []
        try:
            print(files[x+10])
        except:
            print("breaking")
            break
        for z in range(5):
            f = open(os.getcwd() + "/" + str(files[x+z]), "r")
            tmp_store = json.load(f)["people"]
            if len(tmp_store) < 1:
                continue
            tmp_arr.extend(tmp_store[0]["pose_keypoints_2d"])
        data.append(tmp_arr)
        f = open(os.getcwd() + "/" + str(files[x + 10]), "r")

        tmp_store = json.load(f)["people"]
        if len(tmp_store) < 1:
            continue
        labels.append(tmp_store[0]["pose_keypoints_2d"])
out_data = pandas.DataFrame(data)
out_labels = pandas.DataFrame(labels)
os.chdir(origin)
out_data.to_csv("dataframe_data.csv",index=False)
out_labels.to_csv("dataframe_labels.csv", index=False)


print("done in", (time() - start_time), " seconds")



