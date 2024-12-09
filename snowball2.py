import pyrealsense2 as rs
import numpy as np
import cv2
import time

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()



# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)

# Enable some onboard filters on the realsense cam for smoothing
# decimation, HDR Merge, Depth to Disparity, Spatial, Temporal, Hole filling, Disparity to depth

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline_profile = pipeline.start(config)

# toss out 10 frames or so for the camera to self-calibrate
for _ in range(10):
    pipeline.wait_for_frames()

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
blob_params = cv2.SimpleBlobDetector_Params()
blob_params.filterByColor = True
blob_params.filterByArea = True
blob_params.minArea = 100
blob_params.maxArea = 500
blob_params.filterByCircularity = True
blob_params.minCircularity = 0.1
blob_params.filterByConvexity = True
blob_params.minConvexity = 0.5
blob_params.filterByInertia = True
blob_params.minInertiaRatio = 0.01

blob = cv2.SimpleBlobDetector_create(blob_params)

dec_filter = rs.decimation_filter()     # Decimation   - reduces depth frame density
spat_filter = rs.spatial_filter()       # Spatial      - edge-preserving spatial smoothing
temp_filter = rs.temporal_filter()      # Temporal     - reduces temporal noise
hole_filter = rs.hole_filling_filter()  # hole filling - fills gaps caused by the offset of the two cameras


def apply_filters(depth_frame):
    filtered = depth_frame
    # filtered = dec_filter.process(filtered)
    filtered = spat_filter.process(filtered)
    filtered = temp_filter.process(filtered)
    filtered = hole_filter.process(filtered)
    return filtered


last_time = time.time()
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame = apply_filters(depth_frame)
        # Convert to numpy array to display
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image[:, 40:640-20]

        # Scale mm to 0->255
        max_dist_mm = 12 * 25.4
        scaled_image = cv2.convertScaleAbs(depth_image, alpha=8/max_dist_mm)

        cv2.namedWindow('snowball nobg', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('snowball nobg', scaled_image)

        # Background subtract
        fgbg.apply(scaled_image, scaled_image)


        # Apply color for display only
        depth_colormap = scaled_image #cv2.applyColorMap(scaled_image, cv2.COLORMAP_RAINBOW)

        # Show images
        now = time.time()
        dt = now - last_time
        last_time = now

        cv2.namedWindow('snowball', cv2.WINDOW_AUTOSIZE)
        cv2.setWindowTitle('snowball', f'Snowball! (at {round(1/dt)} fps)')
        cv2.imshow('snowball', depth_colormap)

        cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
        color_image = np.asanyarray(frames.get_color_frame().get_data())
        cv2.imshow('camera', color_image)

        # Blob detection
        snowballs = blob.detect(color_image)
        im_with_keypoints = cv2.drawKeypoints(color_image, snowballs, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.namedWindow('found snowballs', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('found snowballs', im_with_keypoints)

        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()