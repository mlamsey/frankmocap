import numpy as np
import cv2

from lamsey.smpl_dict import SMPLDict
from lamsey.lamsey_fmc_tools import fmc_predict, fmc_render

key_frames = [350, 1230]

frame_number = key_frames[0]
cap = cv2.VideoCapture("/nethome/mlamsey3/Documents/data/sws/sws_personalized_matt_0.MP4")

# get total number of frames
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print("Total number of frames: " + str(int(total_frames)))

# check for valid frame number
if frame_number >= 0 & frame_number <= total_frames:
    # set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

ret, frame = cap.read()
bbox_list, prediction_list = fmc_predict(frame)

prediction = prediction_list[0]
pose = prediction["pred_body_pose"][0]
shape = prediction["pred_betas"][0]
joint_vectors = np.array([pose[3*i:3*i+3] for i in range(24)])
joint_rotation_magnitudes = [np.linalg.norm(vector) for vector in joint_vectors]
print(joint_rotation_magnitudes)

# prediction_image = fmc_render(frame, bbox_list, prediction_list)
# cv2.imshow("Video", prediction_image)
# cv2.waitKey()
