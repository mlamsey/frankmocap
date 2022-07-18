import numpy as np
import cv2
import pickle as pkl

from lamsey.smpl_dict import SMPLDict
from lamsey.lamsey_fmc_tools import fmc_predict, fmc_render

key_frames = [350, 1230]

cap = cv2.VideoCapture("/nethome/mlamsey3/Documents/data/sws/sws_personalized_matt_0.MP4")

# get total number of frames
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total number of frames: " + str(int(total_frames)))

poses = []
shapes = []
valid_frames = []

for i in range(total_frames):
    if i % 100 == 0:
        print(i)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()

    if ret:
        bbox_list, prediction_list = fmc_predict(frame)
        if len(bbox_list) > 0:
            prediction = prediction_list[0]
            pose = prediction["pred_body_pose"][0]
            shape = prediction["pred_betas"][0]
            poses.append(pose)
            shapes.append(shape)
            valid_frames.append(i)
        else:
            print("No body found in frame " + str(i))
    else:
        print("Error at frame " + str(i))
    # joint_vectors = np.array([pose[3*i:3*i+3] for i in range(24)])
    # joint_rotation_magnitudes = [np.linalg.norm(vector) for vector in joint_vectors]

f = open("/nethome/mlamsey3/Documents/data/sws/sws_personalized_matt_0_poses.pickle", "wb")
pkl.dump(poses, f)
f.close()

f = open("/nethome/mlamsey3/Documents/data/sws/sws_personalized_matt_0_betas.pickle", "wb")
pkl.dump(shapes, f)
f.close()

f = open("/nethome/mlamsey3/Documents/data/sws/sws_personalized_matt_0_frame_ids.pickle", "wb")
pkl.dump(valid_frames, f)
f.close()

# prediction_image = fmc_render(frame, bbox_list, prediction_list)
# cv2.imshow("Video", prediction_image)
# cv2.waitKey()
