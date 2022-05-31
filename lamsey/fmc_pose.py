# Lamsey 2022
# Analyzes raw images and extracts + saves pose data
image_folder = '/home/hello-robot/test2/'

# imports
import numpy as np
import cv2

from .realsense_fmc import predict_pose
import os

from bodymocap.body_mocap_api import BodyMocap

import torch

# fmc config
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
use_smplx = False
checkpoint_path = './extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt'
smpl_dir = './extra_data/smpl/'
body_mocap = BodyMocap(checkpoint_path, smpl_dir, device, use_smplx)

# main
if __name__ == '__main__':
    smpl_verts = []
    body_poses = []
    betas = []

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    for image in images:
        print(image)
        img = cv2.imread(os.path.join(image_folder, image))
        pose, body_bbox_list = predict_pose(img)

        # Body Pose Regression
        pred_vertices_smpl = np.zeros([6890, 3])
        pred_body_pose = np.zeros([1, 72])
        pred_betas = np.zeros([1, 10])

        if body_bbox_list is not None:
            bbox_size = [(x[2] * x[3]) for x in body_bbox_list]
            idx_big2small = np.argsort(bbox_size)[::-1]
            body_bbox_list = [body_bbox_list[i] for i in idx_big2small]
            pred_output_list = body_mocap.regress(img, body_bbox_list)

            # extract
            out = pred_output_list[0]
            pred_vertices_smpl = out["pred_vertices_smpl"]
            pred_body_pose = out["pred_body_pose"]
            pred_betas = out["pred_betas"]

        smpl_verts.append(np.array(pred_vertices_smpl))
        body_poses.append(np.array(pred_body_pose))
        betas.append(np.array(pred_betas))

    np.save("smpl_verts.npy", np.array(smpl_verts))
    np.save("body_poses.npy", np.array(body_poses))
    np.save("betas.npy", np.array(betas))
