# Lamsey 2022
# Script for analyzing smpl outputs (living doc)

import numpy as np
import matplotlib.pyplot as plt
# import smpl_dict

pwd = "/nethome/mlamsey3/HRL/experiments/sws_fmc/walking_test/rendered/"

for i in range(8):
    file_name = pwd + "key_frame_" + str(i)
    smpl_verts = np.load(file_name + "_smpl_verts.npy")
    body_poses = np.load(file_name + "_body_poses.npy")
    betas = np.load(file_name + "_betas.npy")
    print(betas)


