import pickle as pkl
import matplotlib.pyplot as plt
import cv2
import smplx
import torch
import numpy as np
from os.path import exists

from lamsey.lamsey_fmc_tools import fmc_predict, fmc_render
from lamsey.smpl_dict import SMPLDict

sd = SMPLDict()


with open("/nethome/mlamsey3/Documents/data/sws/sws_personalized_matt_0_poses.pickle", "rb") as f:
    poses = pkl.load(f)

with open("/nethome/mlamsey3/Documents/data/sws/sws_personalized_matt_0_betas.pickle", "rb") as f:
    betas = pkl.load(f)

with open("/nethome/mlamsey3/Documents/data/sws/sws_personalized_matt_0_frame_ids.pickle", "rb") as f:
    valid_frames = pkl.load(f)


def scatter_betas():
    for i in range(len(betas[0])):
        b = [beta[i] for beta in betas]
        marker_sizes = [1 for _ in range(len(betas))]
        plt.scatter(valid_frames, b, s=marker_sizes)

    plt.show()


def render_subset():
    video_path = "/nethome/mlamsey3/Documents/data/sws/sws_personalized_matt_0.MP4"
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    bbox_list, prediction_list = fmc_predict(frame)
    prediction_image = fmc_render(frame, bbox_list, prediction_list)
    height, width, layers = prediction_image.shape
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = "/nethome/mlamsey3/Documents/data/sws/sws_personalized_matt_0_fmc.avi"
    if exists(out):
        print("file already exists, returning.")
        return

    out_video = cv2.VideoWriter(out, fourcc=0, fps=fps, frameSize=(width, height))

    start = 1000
    stop = 1500
    indices = [start + i for i in range(stop - start)]
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        bbox_list, prediction_list = fmc_predict(frame)
        prediction_image = fmc_render(frame, bbox_list, prediction_list)
        # cv2.imshow("Video", prediction_image)
        out_video.write(prediction_image)

    cv2.destroyAllWindows()
    out_video.release()


def analyze_key_points():
    key_frame = 1220

    # smplx setup
    # model_path = "models"
    model_path = "extra_data"
    gender = "male"

    body_model = smplx.create(model_path=model_path,
                              model_type='smplx',
                              gender=gender,
                              use_pca=False,
                              batch_size=1)

    beta = torch.tensor(betas[key_frame]).reshape(1, 10)
    # pose = torch.tensor(poses[key_frame][9:]).reshape(1, 63)
    out = body_model.forward(betas=beta)
    joints = out["joints"][0].detach().numpy()
    x = [j[0] for j in joints][:22]
    y = [j[1] for j in joints][:22]
    z = [j[2] for j in joints][:22]

    right_shoulder = sd.joint_dict["right_shoulder"]
    right_wrist = sd.joint_dict["right_wrist"]
    right_arm_length = np.linalg.norm(joints[right_wrist] - joints[right_shoulder])
    print("Right arm length: " + str(right_arm_length))

    # BELOW: plotting code
    ax = plt.axes(projection='3d')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1.5, 0.5])
    ax.set_zlim([-1, 1])
    #
    # x_ = []
    # y_ = []
    # z_ = []
    # start = 1000
    # stop = 1500
    # indices = [start + i for i in range(stop - start)]
    # for i in indices:
    #     beta = torch.tensor(betas[i]).reshape(1, 10)
    #     pose = torch.tensor(poses[i][9:]).reshape(1, 63)
    #     out = body_model.forward(betas=beta, body_pose=pose)
    #     joints = out["joints"][0].detach().numpy()
    #     x = [j[0] for j in joints][:24]
    #     y = [j[1] for j in joints][:24]
    #     z = [j[2] for j in joints][:24]
    #
    #     x_.append(x)
    #     y_.append(y)
    #     z_.append(z)
    #
    #     ax.scatter3D(x, y, z)
    #     plt.show()
    #     cv2.waitKey()
    #
    # # target_joint = sd.joint_dict["left_palm"]
    # # plt_x = [temp[target_joint] for temp in x_]
    # # plt_y = [temp[target_joint] for temp in y_]
    # # plt_z = [temp[target_joint] for temp in z_]
    #
    # # ax.plot3D(plt_x, plt_y, plt_z)
    #

    ax.scatter3D(x, y, z)
    ax.scatter3D(*[j for j in joints[right_wrist]])
    ax.scatter3D(*[j for j in joints[right_shoulder]])
    plt.legend(["body", "right wrist", "right shoulder"])
    plt.show()


if __name__ == '__main__':
    # scatter_betas()
    # render_subset()
    analyze_key_points()
