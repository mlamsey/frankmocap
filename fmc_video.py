# imports
import numpy as np
import cv2

# fmc
from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
import mocap_utils.demo_utils as demo_utils
from renderer.visualizer import Visualizer

import torch

# main
smpl_verts = []
body_poses = []
betas = []

if __name__ == '__main__':
    image_folder = '/home/hello-robot/test/'
    render_folder = 'rendered/'
    # video_name = 'rendered.avi'
    # fps = 25

    from realsense_fmc import predict_pose, render_prediction, body_mocap
    import os

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # fourcc = 0
    # video = cv2.VideoWriter(os.path.join(image_folder, video_name), fourcc, fps=fps, frameSize=(width,height))

    for image in images:
        print(image)
        img = cv2.imread(os.path.join(image_folder, image))
        pose, bbox = predict_pose(img)
        pred_img = render_prediction(bbox, img)
        cv2.imwrite(os.path.join(image_folder, render_folder, image), pred_img)
        # video.write(pred_img)

        # Body Pose Regression
        pred_vertices_smpl = np.zeros([6890, 3])
        pred_body_pose = np.zeros([1, 72])
        pred_betas = np.zeros([1, 10])
        body_bbox_list = bbox
        if body_bbox_list is not None:
            bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
            idx_big2small = np.argsort(bbox_size)[::-1]
            body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
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
