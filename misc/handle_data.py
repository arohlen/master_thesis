import h5py
import numpy as np
import cv2
from PIL import Image
from torchvision import datasets, models, transforms
import math
import pickle

filename = "D:\Shetty_data\data_labels\data_labels.h5"

with h5py.File(filename, "r") as f:

    # Get all the labels
    all_sat_LLAHTR = list(f["all_sat_LLAHTR"])
    all_uav_LLAHTR = list(f["all_uav_LLAHTR"])
    all_uav_xyzHTR = list(f["all_uav_xyzHTR"])
    match_array_40 = np.array(list(f["match_array_40"]))
    sat_paths = list(f["sat300_image_paths"])
    uav_paths = list(f["uav_image_paths"])


class_count = [0]*64

training_indices = []
validation_indices = []

for i in range(len(all_uav_xyzHTR)):

    x = all_uav_xyzHTR[i][0]
    y = all_uav_xyzHTR[i][1]

    if x >= 200:
        x = 199
    elif x <= -200:
        x = -199

    if y >= 200:
        y = 199
    elif y <= -200:
        y = -199


    x_grid = math.floor(4 + (x / 50)) if x > 0 else math.ceil(3 + (x / 50))
    y_grid = math.floor(4 - (y / 50)) if y > 0 else math.floor(4 - (y / 50))

    pos = 8*y_grid + x_grid

    if class_count[pos] < 10 and pos not in [0,7,56,63]:
        class_count[pos] += 1
        validation_indices.append(i)

    else:
        training_indices.append(i)

import random
import math

uav_training_paths = []
sat_training_paths = []
uav_training_paths_aug = []
sat_training_paths_aug = []

grid_labels_training = []
z_labels_training = []

uav_grid_aug = []
uav_z_aug = []

for i in range(int(len(training_indices))):

    x = all_uav_xyzHTR[training_indices[i]][0]
    y = all_uav_xyzHTR[training_indices[i]][1]

    if x >= 200:
        x = 199
    elif x <= -200:
        x = -199

    if y >= 200:
        y = 199
    elif y <= -200:
        y = -199

    
    x90 = y
    y90 = -x

    x180 = -x
    y180 = -y

    x270 = -y
    y270 = x


    x_grid = math.floor(4 + (x / 50)) if x > 0 else math.ceil(3 + (x / 50))
    y_grid = math.floor(4 - (y / 50)) if y > 0 else math.floor(4 - (y / 50))

    pos = 8*y_grid + x_grid

    x_grid90 = math.floor(4 + (x90 / 50)) if x90 > 0 else math.ceil(3 + (x90 / 50))
    y_grid90 = math.floor(4 - (y90 / 50)) if y90 > 0 else math.floor(4 - (y90 / 50))

    x_grid180 = math.floor(4 + (x180 / 50)) if x180 > 0 else math.ceil(3 + (x180 / 50))
    y_grid180 = math.floor(4 - (y180 / 50)) if y180 > 0 else math.floor(4 - (y180 / 50))

    x_grid270 = math.floor(4 + (x270 / 50)) if x270 > 0 else math.ceil(3 + (x270 / 50))
    y_grid270 = math.floor(4 - (y270 / 50)) if y270 > 0 else math.floor(4 - (y270 / 50))

    pos90 = 8*y_grid90 + x_grid90
    pos180 = 8*y_grid180 + x_grid180
    pos270 = 8*y_grid270 + x_grid270

    uav_grid_aug.append(pos90)
    uav_grid_aug.append(pos180)
    uav_grid_aug.append(pos270)

  
    grid_labels_training.append(pos)
    uav_training_paths.append(uav_paths[training_indices[i]].decode("utf-8"))
    sat_training_paths.append(sat_paths[training_indices[i]].decode("utf-8"))

    z_labels_training.append(all_uav_xyzHTR[training_indices[i]][2]/50)
    uav_z_aug.append(all_uav_xyzHTR[training_indices[i]][2]/50)
    uav_z_aug.append(all_uav_xyzHTR[training_indices[i]][2]/50)
    uav_z_aug.append(all_uav_xyzHTR[training_indices[i]][2]/50)

    for j in range(1,4):
        uav_training_paths_aug.append(uav_paths[training_indices[i]].decode("utf-8")[:-4]+"_{}".format(j))
        sat_training_paths_aug.append(sat_paths[training_indices[i]].decode("utf-8")[:-4]+"_{}".format(j))


z_labels_training += uav_z_aug
grid_labels_training += uav_grid_aug
uav_training_paths += uav_training_paths_aug
sat_training_paths += sat_training_paths_aug

grid_labels_validation = []
z_labels_validation = []

uav_validation_paths = []
sat_validation_paths = []

for i in range(int(len(validation_indices))):

    x = all_uav_xyzHTR[validation_indices[i]][0]
    y = all_uav_xyzHTR[validation_indices[i]][1]

    if x >= 200:
        x = 199
    elif x <= -200:
        x = -199

    if y >= 200:
        y = 199
    elif y <= -200:
        y = -199

    
    x_grid = math.floor(4 + (x / 50)) if x > 0 else math.ceil(3 + (x / 50))
    y_grid = math.floor(4 - (y / 50)) if y > 0 else math.floor(4 - (y / 50))

    pos = 8*y_grid + x_grid

    grid_labels_validation.append(pos)

    z_labels_validation.append(all_uav_xyzHTR[validation_indices[i]][2]/50)

    uav_validation_paths.append(uav_paths[validation_indices[i]].decode("utf-8"))
    sat_validation_paths.append(uav_paths[validation_indices[i]].decode("utf-8"))

labels = {"uav_training_paths":uav_training_paths,"sat_training_paths":sat_training_paths,\
    "uav_validation_paths":uav_validation_paths,"sat_validation_paths":sat_validation_paths,\
    "grid_labels_training":grid_labels_training,"grid_labels_validation":grid_labels_validation,\
    "z_labels_training":z_labels_training,"z_labels_validation":z_labels_validation}

with open('D:/Shetty_data/train/training_labels.pickle', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
