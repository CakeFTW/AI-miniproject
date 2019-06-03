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
    print("processing:", path)
    PeopleCounter = 0
    NoPeopleCounter = 0
    NotFullBodyCounter = 0
    for x in range(5,len(files)-5):
        tmp_arr = []
        people = True
        for z in range(5):
            f = open(os.getcwd() + "/" + str(files[x-z]), "r")
            tmp_store = json.load(f)["people"]
            
            if len(tmp_store) < 1:
                people = False
                continue
            tmp_arr.extend(tmp_store[0]["pose_keypoints_2d"])
        f = open(os.getcwd() + "/" + str(files[x + 5]), "r")
        tmp_store = json.load(f)["people"]
        if len(tmp_store) < 1:
            people = False
            continue
        if people:
            if 0 in tmp_arr or 0 in tmp_store:
                NotFullBodyCounter += 1
            else:
                labels.append(tmp_store[0]["pose_keypoints_2d"])
                data.append(tmp_arr)
                PeopleCounter += 1
        else:
            NoPeopleCounter += 1
    print("done found ", PeopleCounter, "people and ",NotFullBodyCounter, " frames lacked features and ," , NoPeopleCounter, "frames missing people")   


out_data = pandas.DataFrame(data)
out_labels = pandas.DataFrame(labels)
os.chdir(origin)
out_data.to_csv("dataframe_data.csv",index=False)
out_labels.to_csv("dataframe_labels.csv", index=False)


print("done in", (time() - start_time), " seconds")



