import numpy as np
import cv2
import os
import torch

# fmc
from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
import mocap_utils.demo_utils as demo_utils
from renderer.visualizer import Visualizer


# fmc setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
body_bbox_detector = BodyPoseEstimator()
use_smplx = False
checkpoint_path = './extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt'
smpl_dir = './extra_data/smpl/'
renderer_type = "opengl"
body_mocap = BodyMocap(checkpoint_path, smpl_dir, device, use_smplx)
visualizer = Visualizer(renderer_type)


def fmc_output_extract(pred_output_list, i=0):
    out = pred_output_list[i]
    pred_vertices_smpl = out["pred_vertices_smpl"]
    pred_body_pose = out["pred_body_pose"]
    pred_betas = out["pred_betas"]

    return pred_vertices_smpl, pred_body_pose, pred_betas


def fmc_predict(img):
    # match fmc names
    img_original_bgr = img

    # from demo script
    body_pose_list, body_bbox_list = body_bbox_detector.detect_body_pose(
        img_original_bgr)

    if len(body_bbox_list) < 1:
        print(f"No body detected.")
        return [], []

    pred_output_list = body_mocap.regress(img, body_bbox_list)

    return body_bbox_list, pred_output_list


def fmc_render(img, body_bbox_list, pred_output_list):
    # match fmc names
    img_original_bgr = img

    # extract mesh for rendering (vertices in image space and faces) from pred_output_list
    pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

    # visualization
    res_img = visualizer.visualize(
        img_original_bgr,
        pred_mesh_list=pred_mesh_list,
        body_bbox_list=body_bbox_list)

    return res_img


def fmc_folder(image_folder, render_folder):
    pass


def images_to_video(image_folder, video_name='rendered.avi', fps=25):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(os.path.join(image_folder, video_name), fourcc=0, fps=fps, frameSize=(width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()
