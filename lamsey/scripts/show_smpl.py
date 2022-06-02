import numpy as np
import smplx
import trimesh
import torch
import lamsey.SMPLXProcessor as sp

body_model = smplx.create(model_path="/nethome/mlamsey3/HRL/frankmocap/extra_data/smpl/smpl_neutral_lbs_10_207_0_v1.0.0.pkl",
                          model_type="smpl",
                          gender="neutral",
                          use_pca=False,
                          batch_size=1)

pwd = "/nethome/mlamsey3/HRL/experiments/sws_fmc/walking_test/rendered/"
betas = torch.tensor(np.load(pwd + "key_frame_0_betas.npy"))

model_output = body_model(betas=betas, return_verts=True)
vertices, faces, joints = sp.extract_model_output(body_model, model_output)

face_colors = np.ones([body_model.faces.shape[0], 4]) * [0.7, 0.7, 0.7, 0.8]

# generate trimesh
m = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors)

# create scene
scene = m.scene()
scene.show()