import pyrealsense2 as rs
import numpy as np
import cv2
import json
import threading

import asyncio
import socketio

from uvicorn import Config, Server

import socketio
import time

sio = socketio.AsyncServer(async_mode='asgi')
static_files = {'': './static'}
app = socketio.ASGIApp(sio, static_files=static_files)


@sio.event
async def connect(sid, environ):
    print("connect ", sid)


@sio.event
async def disconnect(sid):
    print("disconnect ", sid)


@sio.event
async def message(sid, data):
    await sio.emit('message', data, broadcast=True, include_self=False)


class FlyingSnowball:

    def __init__(self, x, y, distance_to_wall, size=None):
        self.x = x
        self.y = y
        self.distance_to_wall = distance_to_wall
        self.size = size

        self.hit_x = None
        self.hit_y = None

        self.hit_wall = False

        self.id = None

        self.last_updated = time.time()

        self.history = [(x, y, distance_to_wall, self.last_updated)]
        self.needs_prediction = False

    def merge(self, other):
        """
        Take other snowball data and merge it with our own data
        """
        self.history.append((self.x, self.y, self.distance_to_wall, self.last_updated))  # stuff our old data into history
        self.x = other.x
        self.y = other.y
        self.distance_to_wall = other.distance_to_wall
        self.last_updated = time.time()


class SnowballTracker:

    def __init__(self, merge_threshold_x_px=50, merge_threshold_y_px=100, stale_timeout_secs=1.0):
        self.snowballs: [FlyingSnowball] = []
        self.snowball_id = 1
        self.merge_threshold_x = merge_threshold_x_px
        self.merge_threshold_y = merge_threshold_y_px
        self.stale_timeout_secs = stale_timeout_secs

    def merge_snowballs(self, found_snowballs: [FlyingSnowball]):
        """
        Go through each snowball we're given and either add it to our list or update
        a snowball already in our list.
        """
        # Mark all our current snowballs as needing prediction
        for s in self.snowballs:
            s.needs_prediction = True

        for f in found_snowballs:
            for s in self.snowballs:
                if abs(s.x - f.x) < self.merge_threshold_x and abs(s.y - f.y) < self.merge_threshold_y:
                    s.merge(f)
                    s.needs_prediction = False
                    break
            else:
                f.id = self.snowball_id
                self.snowball_id += 1
                self.snowballs.append(f)

    def get_wall_collisions(self):
        """
        Go through each snowball and see if we're close enough to the wall to mark a collision
        Any snowballs returned by this function will have their hit_wall set to True
        """
        hits = []
        for s in self.snowballs:
            hit = False
            if s.hit_wall:
                continue
            hit = s.distance_to_wall < 500 and len(s.history) >= 2 and s.history[-1][2] < s.history[-2][2]
            # hit = hit or (s.distance_to_wall < 500 and s.y < 200)
            if hit:
                hits.append(s)
                s.hit_wall = True
                s.hit_x = s.x
                s.hit_y = s.y
        return hits

    def purge_old_snowballs(self):
        now = time.time()
        old = len(self.snowballs)
        self.snowballs = [s for s in self.snowballs if now - s.last_updated < self.stale_timeout_secs]
        new = len(self.snowballs)

    def debug_print(self):
        now = time.time()
        for s in self.snowballs:
            print(f"{'HIT ' if s.hit_wall else ''}{s.id}: ({s.x}, {s.y}), d: {s.distance_to_wall}, age: {now - s.last_updated}")
        # sys.stdout.flush()

    def predict_snowballs(self):
        """
        Takes all snowballs that haven't been updated in the last merge_snowballs call and predicts their new position
        """
        now = time.time()
        for s in self.snowballs:
            if s.needs_prediction:
                s.needs_prediction = False
                if len(s.history) < 2:
                    continue

                # Grab our last real history entry
                # compare our last X to current X, last Y to Y, last D to D, and compute linear velocities.
                # update based on our time delta

                # our previous position in time (our x, y, etc. hold the current data)
                x1, y1, d1, t1 = s.history[-1]
                x2, y2, d2, t2 = s.history[-2]

                dt = t1-t2
                if dt == 0:
                    # idk
                    continue
                vx = (x1 - x2) / dt
                vy = (y1 - y2) / dt
                vd = (d1 - d2) / dt

                # Update our current based on the time passed
                time_since_real_update = now - s.last_updated
                s.x += vx * time_since_real_update
                s.y += vy * time_since_real_update
                s.distance_to_wall += vd * time_since_real_update


class SnowballFinder:

    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)

        self.cam_width = 640
        self.cam_height = 480
        self.cam_left_trim = 170
        self.cam_right_trim = 160
        self.cam_top_trim = 100
        self.cam_bottom_trim = 200

        self.config.enable_stream(rs.stream.depth, self.cam_width, self.cam_height, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, self.cam_width, self.cam_height, rs.format.bgr8, 30)

        # Start streaming
        pipeline_profile = self.pipeline.start(self.config)

        # Set our IR emitter to be FULL POWAH
        self.device = pipeline_profile.get_device().query_sensors()[0].set_option(rs.option.laser_power, 360)


        # toss out 10 frames or so for the camera to self-calibrate
        for _ in range(100):
            self.pipeline.wait_for_frames()


        # Enable some onboard filters on the realsense cam for smoothing
        # decimation, HDR Merge, Depth to Disparity, Spatial, Temporal, Hole filling, Disparity to depth
        self.dec_filter = rs.decimation_filter()     # Decimation   - reduces depth frame density
        self.spat_filter = rs.spatial_filter()       # Spatial      - edge-preserving spatial smoothing
        self.temp_filter = rs.temporal_filter()      # Temporal     - reduces temporal noise
        self.hole_filter = rs.hole_filling_filter()  # hole filling - fills gaps caused by the offset of the two cameras

        # Make a blob detector for our snowballs
        # https://learnopencv.com/blob-detection-using-opencv-python-c/
        params = cv2.SimpleBlobDetector_Params()
        # Thresholding
        params.minThreshold = 0  # Adjust this based on your image intensity
        params.maxThreshold = 255
        # Filter by Area
        params.filterByArea = True
        params.minArea = 10 * 10  # Adjust this based on the size of your ball
        params.maxArea = 50 * 50
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.5
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.2
        # Create a detector with the parameters
        self.detector = cv2.SimpleBlobDetector_create(params)

        self.last_debug_time = time.time()
        self.last_time = time.time()
        self.mask = None

    def apply_filters(self, depth_frame):
        filtered = depth_frame
        # filtered = dec_filter.process(filtered)
        filtered = self.spat_filter.process(filtered)
        filtered = self.temp_filter.process(filtered)
        filtered = self.hole_filter.process(filtered)
        return filtered

    def find_snowballs(self, cached_hits):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_frame = self.apply_filters(depth_frame)
        # Convert to numpy array to display
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = depth_image[self.cam_top_trim:self.cam_height-self.cam_bottom_trim, self.cam_left_trim:self.cam_width-self.cam_right_trim]

        # Remove our background mask from our captured image
        if self.mask is None:
            self.mask = depth_image
        delta_image = cv2.subtract(self.mask, depth_image)

        # Scale mm to 0->255
        # max_dist_mm = 12 * 25.4
        max_dist_mm = 4 * 25.4
        scaled_image = cv2.convertScaleAbs(delta_image, alpha=8/max_dist_mm)

        # Find snowballs
        inverted = cv2.bitwise_not(scaled_image)
        keypoints = self.detector.detect(inverted)
        im_with_keypoints = cv2.drawKeypoints(inverted, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        snowballs = []
        for k in keypoints:
            x = int(k.pt[0])
            y = int(k.pt[1])
            # print(f"mask: {self.mask[y][x]}")
            # print(f"depth_image: {depth_image[y][x]}")
            # print(f"delta_image: {delta_image[y][x]}")
            snowballs.append(FlyingSnowball(x, y, int(delta_image[y][x]), k.size))

        # Show images
        now = time.time()
        self.dt = now - self.last_time
        self.last_time = now

        if True or now - self.last_debug_time > 1:
            self.last_debug_time = now

            cv2.namedWindow('snowball', cv2.WINDOW_AUTOSIZE)
            cv2.setWindowTitle('snowball', f'Snowball! (at {round(1/self.dt)} fps)')
            cv2.imshow('snowball', im_with_keypoints)

            cv2.namedWindow('camera', cv2.WINDOW_AUTOSIZE)
            color_image = np.asanyarray(frames.get_color_frame().get_data())
            x1 = self.cam_left_trim
            y1 = self.cam_top_trim
            x2 = self.cam_width - self.cam_right_trim
            y2 = self.cam_height - self.cam_bottom_trim
            # Draw a red rectangle
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for h in cached_hits:
               cv2.circle(color_image, (int(h.hit_x)+40, int(h.hit_y)), 5, (0, 0, 255), -1)
            cv2.drawKeypoints(color_image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow('camera', color_image)
            cv2.waitKey(1)

        fps = round(1 / self.dt)
        return snowballs, fps

    def get_ratio(self, x, y):
        """
        Given an x, y position within the trimmed image's region
        return the ratio of x and y across the image dimensions.

        Basically we might hit at (100, 200) but we trim the left 100 pixels.
        In this case we'd return (0, 0.25) or whatever
        """
        # without trim the math is:
        x_ratio = x / self.cam_width
        y_ratio = y / self.cam_height

        # but since we have trim the math is
        # x_ratio = (x - self.cam_left_trim) / (self.cam_width - self.cam_left_trim - self.cam_right_trim)
        # y_ratio = (y - self.cam_top_trim) / (self.cam_height - self.cam_top_trim - self.cam_bottom_trim)

        x_ratio = ((x - self.cam_left_trim) / (self.cam_width - self.cam_right_trim - self.cam_left_trim))
        y_ratio = ((y - self.cam_top_trim) / (self.cam_height - self.cam_bottom_trim - self.cam_top_trim))


        return (x_ratio, y_ratio)

async def run_snowball_game():
    finder = SnowballFinder()
    tracker = SnowballTracker(merge_threshold_x_px=50, merge_threshold_y_px=150, stale_timeout_secs=0.2)

    cached_hits = []
    last_print_time = time.time()

    while True:
        snowballs, fps = finder.find_snowballs(cached_hits)
        tracker.merge_snowballs(snowballs)
        tracker.predict_snowballs()
        tracker.purge_old_snowballs()
        tracker.debug_print()
        hits = tracker.get_wall_collisions()

        if hits:
            for s in hits:
                cached_hits.append(s)
                xr, yr = finder.get_ratio(s.hit_x, s.hit_y)
                dts = {"xr": xr, "yr": yr, "x_px": int(s.hit_x), "y_px": int(s.hit_y)}
                print(f"Snowball hit at: {dts}")
                await sio.emit('message', dts)

        now = time.time()
        if now - last_print_time > 5:
            print( f'Snowball! (at {fps} fps)')
            last_print_time = now
            await sio.emit('heartbeat', {'fps': fps})

        await asyncio.sleep(0)


async def main():
    asyncio.create_task(run_snowball_game())
    config = Config(app, host="0.0.0.0", port=5000)
    server = Server(config)
    await server.serve()

if __name__ == '__main__':
    asyncio.run(main())

