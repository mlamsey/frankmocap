import numpy as np
import smplx
import trimesh
import torch
import lamsey.SMPLXProcessor as sp
from lamsey.smpl_dict import SMPLDict

body_model = smplx.create(model_path="/nethome/mlamsey3/HRL/frankmocap/extra_data/smpl/smpl_neutral_lbs_10_207_0_v1.0.0.pkl",
                          model_type="smpl",
                          gender="neutral",
                          use_pca=False,
                          batch_size=1)

pwd = "/nethome/mlamsey3/HRL/experiments/sws_fmc/walking_test/rendered/"
i = 0
for i in range(8):
    file_name = "key_frame_" + str(i) + "_betas.npy"
    betas = torch.tensor(np.load(pwd + file_name))

    model_output = body_model(betas=betas, return_verts=True)
    vertices, faces, joints = sp.extract_model_output(body_model, model_output)

    # for entry, id in SMPLDict().joint_dict.items():
        # print(entry + ": " + str(joints[id]))

    sd = SMPLDict()
    head_id = sd.joint_dict["head"]
    left_foot_id = sd.joint_dict["left_foot"]

    height = max(vertices[:, 1]) - min(vertices[:, 1])
    print("Frame " + str(i))
    print("Human Height: {height:.2f}m".format(height=height))
    print("Actual: 1.67m to 1.72m")