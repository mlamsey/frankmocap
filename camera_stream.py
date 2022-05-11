# imports
import pyrealsense2 as rs
import numpy as np
import cv2

# config
resolution_depth = [640, 480]
resolution_color = [640, 480]
fps_color = 30

# setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, resolution_depth[0], resolution_depth[1], rs.format.z16, fps_color) # note that the fps downsampling will be applied later
config.enable_stream(rs.stream.color, resolution_color[0], resolution_color[1], rs.format.bgr8, fps_color)

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

        colorAndDepth_image = np.hstack((color_image, depth_image))

        cv2.namedWindow('Realsense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Realsense', colorAndDepth_image)

        if cv2.waitKey(1) & 0xFF != 255:
            raise KeyboardInterrupt()

    except KeyboardInterrupt:
        break