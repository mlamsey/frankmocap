# imports
import pyrealsense2 as rs
import numpy as np
import cv2

# fmc
from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
import mocap_utils.demo_utils as demo_utils
from renderer.visualizer import Visualizer

import torch

# config
resolution_depth = [640, 480]
resolution_color = [640, 480]
fps_color = 30

# setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, resolution_depth[0], resolution_depth[1], rs.format.z16, fps_color) # note that the fps downsampling will be applied later
config.enable_stream(rs.stream.color, resolution_color[0], resolution_color[1], rs.format.bgr8, fps_color)

# fmc setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
use_cuda = device.type == 'cuda'
body_bbox_detector = BodyPoseEstimator(use_cuda=use_cuda)
use_smplx = False
checkpoint_path = './extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt'
smpl_dir = './extra_data/smpl/'
renderer_type = "opengl"
body_mocap = BodyMocap(checkpoint_path, smpl_dir, device, use_smplx)
visualizer = Visualizer(renderer_type)

# funcs
def predict_pose(img):
    # match fmc names
    img_original_bgr = img

    # from demo script
    body_pose_list, body_bbox_list = body_bbox_detector.detect_body_pose(
                img_original_bgr)
    
    if len(body_bbox_list) < 1: 
        print(f"No body detected.")
        return None, None
        
    return body_pose_list, body_bbox_list

def render_prediction(body_bbox, img):
    if body_bbox is None:
        # return visualizer.visualize(img)
        return np.concatenate((img, img), axis=1)
    # match fmc names
    body_bbox_list = body_bbox
    img_original_bgr = img

    # config [dirty]
    single_person = True

    #Sort the bbox using bbox size 
    # (to make the order as consistent as possible without tracking)
    bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
    idx_big2small = np.argsort(bbox_size)[::-1]
    body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
    if single_person and len(body_bbox_list)>0:
        body_bbox_list = [body_bbox_list[0], ]       

    # Body Pose Regression
    pred_output_list = body_mocap.regress(img_original_bgr, body_bbox_list)
    assert len(body_bbox_list) == len(pred_output_list)

    # extract mesh for rendering (vertices in image space and faces) from pred_output_list
    pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)

    # visualization
    res_img = visualizer.visualize(
        img_original_bgr,
        pred_mesh_list = pred_mesh_list, 
        body_bbox_list = body_bbox_list)
    
    return res_img

# main
if __name__ == '__main__':
    pipeline.start(config)
    while True:
        try:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays.
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=-0.04, beta=255.0), cv2.COLORMAP_HSV)

            color_image = np.moveaxis(color_image, 0, 1)
            color_image = np.fliplr(color_image)
            depth_image = np.moveaxis(depth_image, 0, 1)
            depth_image = np.fliplr(depth_image)

            body_pose, body_bbox = predict_pose(color_image)
            if body_pose is not None:
                res_img = render_prediction(body_bbox, color_image)

                cv2.namedWindow('frankmocap', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('frankmocap', res_img)

            # colorAndDepth_image = np.hstack((color_image, depth_image))

            # cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)
            # cv2.imshow('Realsense', colorAndDepth_image)

            if cv2.waitKey(1) & 0xFF != 255:
                raise KeyboardInterrupt()

        except KeyboardInterrupt:
            break