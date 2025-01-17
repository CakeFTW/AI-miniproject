"This module contains the data_processing scrips"

import numpy as np
import pandas as pd


def remove_confidence_intervals(data:np.ndarray):
    """Sets all key points relative to the center but scale 0-1 using screen dimensions"""
    # if data.shape[1] != 375:
    #     print("wrong dimensions, DUDE! should be 375 MAAAAN")
    #     exit()

    data = np.nan_to_num(data)

    data_nci = np.delete(data,[3*x+2 for x in range(25*5)], 1)
    print([3*x+2 for x in range(25*5)])

    # #remove unsued joints
    # unused_joints = [0,10,13,11,14,15,16,17,18,19,20,21,22,23,24]
    #
    # unused_joints = [2*x for x in unused_joints]
    #
    # unused_joints = unused_joints + [x+1 for x in unused_joints]
    #
    # data_new = np.delete(data_nci, unused_joints, 1)

    #return data_new
    return data_nci


# def circle_scale_all(data : np.ndarray):
#     ln = data.shape[1]
#     out_1 = circle_scale(data.copy())
#     out_2 = circle_scale(data.copy())
#     out_3 = circle_scale(data.copy())
#     out_1 = np.append(out_1,out_2, axis=1)
#     out_1 = np.append(out_1,out_3, axis=1)
#     return out_1
#
# def circle_scale_old(data : np.ndarray, joint = 24):
#     """Sets all key points relative to the center but scale 0-1 using screen dimensions"""
#     if data.shape[1] != 75:
#         print("wrong dimensions, DUDE! should be 75 MAAAAN")
#         exit()
#     #scale according to hips
#     x_index = data[:,joint]
#     y_indey = data[:,joint + 1]
#     for i in range(25):
#         data[:,i*3] = data[:,i*3] - x_index
#         data[:,(i*3)+1] = data[:,(i*3)+1] - y_indey
#
#     #do min-max scaling but using the screen
#     for i in range(25):
#         total = np.sqrt( np.square(data[:,i*3]) + np.square(data[:,(i*3)+1]))
#         data[:,i*3] = data[:,i*3]/total
#         data[:, (i*3)+1] = data[:,(i*3)+1]/total
#     data = np.nan_to_num(data)
#     return data


def relativeToFirstIndex(data : np.ndarray, number_of_frames = 5, nrOfFeatures = 50):
    "Makes each coordinate relative to the first index in an array of length number of frames x 25"

    for x in range(number_of_frames - 1):
        index = x+1
        newdata = data
        for y in range(nrOfFeatures):
            newdata[:,y + nrOfFeatures*index] = data[:, y + nrOfFeatures*index] - data[:,y + nrOfFeatures * x]

    return data


def calcAnglesBody(data : np.ndarray, number_of_frames = 5):
    new_data = np.empty((data.shape[0],15*number_of_frames))
    targets = [[4,2,3],[3,1,2],[7,5,6],[6,1,5],[9,1,8],[10,8,9],[11,9,10],[22,10,11],[12,1,8],[13,8,12],[14,12,13],[19,13,14],[8,0,1],[1,16,0],[1,15,0]]
    counter = 0
    for frame in range(number_of_frames):
        startIndex = frame * 25
        for target in targets:
            new_data[:,counter] = calcAngleJoint(data, target[0]+startIndex, target[1]+startIndex, target[2]+startIndex)
            counter += 1
    
    return np.nan_to_num(new_data)



def dotsProducs(x1,y1,x2,y2):
    "calculates dot products"
    return np.transpose(np.array([x1*x2 + y1*y2 , x1*(-y2) + y1*x2]))

def calcAngleJoint(data,target1, target2, joint):
    """creates vectors from join to vector 1 and 2; normalizes them, then calculates the dot product between v1 * v2 
    and v1  * a vector perpendicular to v2"""
    #calculate the vectors
    vector1 = data[:, target1*2:target1*2+2]
    vector2 = data[:, target2*2:target2*2+2]
    joint = data[:, joint*2:joint*2+2]

    vector1 -= joint
    vector2 -= joint

    #now normalize the vectors
    norm1 = np.sqrt(np.power(vector1[:,0],2) + np.power(vector1[:,1],2))
    norm2 = np.sqrt(np.power(vector2[:,0],2) + np.power(vector2[:,1],2))

    vector1[:,0] = np.divide(vector1[:,0],norm1)
    vector1[:,1] = np.divide(vector1[:,1],norm1)
    vector2[:,0] = np.divide(vector2[:,0],norm2)
    vector2[:,1] = np.divide(vector2[:,1],norm2)
    vector1 = np.nan_to_num(vector1)
    vector2 = np.nan_to_num(vector2)
    angleCosAndSin =  dotsProducs(vector1[:,0],vector1[:,1], vector2[:,0], vector2[:,1])
    angle = 0.5 + 0.5 * angleCosAndSin[:,0] * np.sign(angleCosAndSin[:,1])
    print(angle)
    return angle




# data_set = pd.read_csv("dataframe_data.csv")
# data_set_np = data_set.values
# data_set_np = remove_confidence_intervals(data_set_np)

# dots = calcAnglesBody(data_set_np)

# print(dots.shape)


# data_set_np = angleRelToLastJoint(data_set_np)

# print(data_set_np)
# print(data_set_np.shape)
