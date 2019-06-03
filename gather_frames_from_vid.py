    
import os
import cv2
from time import time, sleep
from sys import argv
import _thread

start_time = time()
try:
    dirs =  os.listdir(argv[1])
except:
    print("no folder provided")

origin = os.getcwd()

for path in dirs:
    os.mkdir(origin + "/" + argv[1] + "/" + path[:-4])
    os.chdir(origin + "/" + argv[1])
    print("processing:", path, " in directory:",os.getcwd())
    cap=cv2.VideoCapture(str(path))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if(total_frames>2000):
        current_frame=int(total_frames/3)
        max_frames=1000
        print("over 2k frames")
    elif((total_frames<=2000) and (total_frames>1000)):
        current_frame=int(total_frames/5)
        max_frames=700
        print("over 1k frames")
    elif(total_frames<=1000 and total_frames>500):
        current_frame=int(total_frames/8)
        max_frames=400
        print("over 0.5k frames")
    else: 
        print("not enough frames")
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    os.chdir(origin + "/" + argv[1] + "/" + path[:-4])
    for i in range(current_frame, max_frames+current_frame):
        res, frame = cap.read()
        if(res==True):
            printed_frame=str(i-current_frame+1)
            file_name=path[:-4]+printed_frame.zfill(8)+".png"
            try:
                cv2.imwrite(file_name, frame)
            except Exception as e:
                print(e)
            print(i, " extracted")
    cap.release()