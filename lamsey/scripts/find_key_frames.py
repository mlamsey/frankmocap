import os
import cv2
import numpy as np
import lamsey.lamsey_fmc_tools as lft

# config
image_folder='/nethome/mlamsey3/HRL/experiments/sws_fmc/walking_test/'
render_folder='rendered/'
test_img_file = image_folder + '00100_30fps.png'
test_img = cv2.imread(test_img_file)

i = 0
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

for image in images:
    img = cv2.imread(image_folder + image)

    # run frankmocap
    body_bbox_list, pred_output_list = lft.fmc_predict(img)
    rendered_image = lft.fmc_render(img, body_bbox_list, pred_output_list)

    if len(pred_output_list) > 0:
        pred_vertices_smpl = pred_output_list[0]["pred_vertices_smpl"]
        pred_body_pose = pred_output_list[0]["pred_body_pose"]
        pred_betas = pred_output_list[0]["pred_betas"]

    # show image
    cv2.imshow("window", rendered_image)

    key = cv2.waitKey()
    if key == 115:  # s
        print("Saving image! Key frame #" + str(i))
        file_name = image_folder + render_folder + "key_frame_" + str(i)
        cv2.imwrite(file_name + ".png", rendered_image)
        np.save(file_name + "_smpl_verts.npy", np.array(pred_vertices_smpl))
        np.save(file_name + "_body_poses.npy", np.array(pred_body_pose))
        np.save(file_name + "_betas.npy", np.array(pred_betas))
        i += 1
    elif key == 113:
        break

