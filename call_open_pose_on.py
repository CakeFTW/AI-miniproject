from os import *
import shutil
from sys import argv
import datetime


def call_openpose_on_folder(path):
    "Use open pose to process all video located in folder, saving all videos in a separate folder"
    out_dir = getcwd()

    try:
        items_in_folder = listdir(path)
    except:
        print("no folder path provided")

    videos_to_process = [x for x in items_in_folder if ".mp4" in x]

    try:
        mkdir(argv[1] + "_poses")
    except:
        print("Output folder already exists, rename it")

    chdir(r"D:\Openpose\openpose-master")

    pose_path = r'build-old\x64\Release\openposedemo.exe'
    

    #pose_flags = r' --display 0 --render_pose 0'
    pose_flags = r""


    for video in videos_to_process:
        pose_media = "--video " + out_dir + "/" + argv[1]+ "/" + video
        output_dest = "--write_json " + out_dir + "/" + argv[1] + "_poses/" +video

        try:
            system("{} {} {} {}".format(pose_path, pose_media, output_dest, pose_flags))
        except:
            print("error trying to start open pose")
            exit()


call_openpose_on_folder(argv[1])





