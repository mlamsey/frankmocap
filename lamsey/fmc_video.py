# Lamsey 2022
# Analyzes raw images and extracts + saves pose data plus video frames
image_folder = '/nethome/mlamsey3/HRL/experiments/sws_fmc/test/'
render_folder = 'rendered/'

# imports
import numpy as np
import cv2
from .realsense_fmc import predict_pose, render_prediction
import os

# fmc
from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
import mocap_utils.demo_utils as demo_utils
from renderer.visualizer import Visualizer

import torch

# video write func
def images_to_video(dir_path, fps=25):
    image_folder = dir_path
    # video_name = 'test_raw.mp4'
    video_name = 'rendered.avi'

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video = cv2.VideoWriter(os.path.join(image_folder, video_name), fourcc, fps=fps, frameSize=(width,height))
    fourcc = 0
    video = cv2.VideoWriter(os.path.join(image_folder, video_name), fourcc, fps=fps, frameSize=(width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# main
if __name__ == '__main__':
    smpl_verts = []
    body_poses = []
    betas = []

    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape


    for image in images:
        # filename for progress in terminal
        print(image)

        # load image, predict pose, save frame to rendered folder
        img = cv2.imread(os.path.join(image_folder, image))
        pose, bbox = predict_pose(img)
        pred_img, pred_output_list = render_prediction(bbox, img)
        cv2.imwrite(os.path.join(image_folder, render_folder, image), pred_img)

        # Save model output
        pred_vertices_smpl = np.zeros([6890, 3])
        pred_body_pose = np.zeros([1, 72])
        pred_betas = np.zeros([1, 10])

        if bbox is not None and pred_output_list is not None:
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

    print("Rendering video... ")
    images_to_video(os.path.join(image_folder, render_folder))
    print("Done!")
