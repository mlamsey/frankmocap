# Lamsey 2022
# Script for analyzing smpl outputs (living doc)

import numpy as np
import matplotlib.pyplot as plt
import smpl_dict

pwd = "/home/hello-robot/data/num/walking/"
smpl_verts = np.load(pwd+"smpl_verts.npy")
body_poses = np.load(pwd+"body_poses.npy")
betas = np.load(pwd+"betas.npy")

# n_frames = betas.shape[0]

# betas = np.squeeze(betas)
# for i in range(betas.shape[0]):
#     plt.plot(betas)

body_poses = np.squeeze(body_poses)
n_frames = body_poses.shape[0]

sd = smpl_dict.SMPLDict()

# for pose in body_poses:
pose = body_poses[200]
for key, _ in sd.joint_dict.items():
    i = sd.get_pose_ids(key)
    axis_angle = pose[i]
    angle = np.linalg.norm(axis_angle)
    direction = axis_angle / angle

    print(key)
    print("Angle: " + str(angle) + " rad")
    print("Direction: " + str(direction))
    print(" ")

# plt.show()