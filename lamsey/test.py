import cv2
import lamsey.lamsey_fmc_tools as lft

# config
image_folder='/nethome/mlamsey3/HRL/experiments/sws_fmc/test/'
render_folder='rendered/'
test_img_file = image_folder + '00100.png'
test_img = cv2.imread(test_img_file)

# run frankmocap
body_bbox_list, pred_output_list = lft.fmc_predict(test_img)
rendered_image = lft.fmc_render(test_img, body_bbox_list, pred_output_list)

# show image
cv2.imshow("window", rendered_image)
cv2.waitKey()
