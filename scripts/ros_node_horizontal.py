#!/usr/bin/env python2.7

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import pickle
from pipeline import DetectionPipeline
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from std_msgs.msg import Bool
import sys
import time
import signal
import os

class LaneDetectionRosNodeHorizontal:
    def __init__(self):
        rospy.init_node('horizontal_line_node', anonymous = True)

        self.isActive = False

        self.sub = rospy.Subscriber('/front_camera/image_raw', Image, self.image_callback, queue_size = 1)
        self.sub_active = rospy.Subscriber('/waypoint_planning/horizontal_line_detection', Bool, self.active_callback, queue_size = 1)
        self.stop_line_distance = rospy.Publisher('/lane_detection/stop_line_distance', Float32, queue_size = 1)

        # Initialize class members
        self.image = None
        self.calibration_data = pickle.load(open("/home/team0/catkin_ws/src/lane_detection/scripts/camera_calibration_data.p", "rb"))
        self.pipeline = DetectionPipeline((1024, 1024), self.calibration_data)

    # Show single image
    def show_single_image(self, name, image):
        cv2.imshow(name, image)
        cv2.waitKey(1)

    # ROS subscriber callback (should we be running or not)
    def active_callback(self, msg):
        if(self.isActive == True and msg.data == False):
            print("Horizontal Line Detection - Deactivating...")
            self.isActive = False
        elif(self.isActive == False and msg.data == True):
            print("Horizontal Line Detection - Activating...")
            self.isActive = True

    # ROS subscriber callback
    def image_callback(self, image_msg):
        bridge = CvBridge()

        # Check if image data exists  
        if image_msg.data is None:
            raise RuntimeError("Null image data read from message.")

        # Convert to cv2 mat type
        image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

        if image is None:
            raise RuntimeError("Decoded image is null")

        self.image = image

    def resize_image(self, img, target_size):
        target_size = float(target_size)
        height = img.shape[0]
        width = img.shape[1]
        if height > width:
            scale_factor = target_size / height
            img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
        else:
            scale_factor = target_size / width
            img = cv2.resize(img, (int(width * scale_factor), int(height * scale_factor)))
        return img

    def apply_h(self, rgb_image):
        """Initiates the filtering process"""

        self.rgb_image = rgb_image

        # separate hls (hue, lightness, and saturation) channels
        self.hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
        self.h = self.hls[:, :, 0]
        self.l = self.hls[:, :, 1]
        self.s = self.hls[:, :, 2]

        # create white color mask
        self.white_color_mask = self.create_white_color_mask_h()

        # make copy?
        filtered_img = self.white_color_mask

        return filtered_img

    def create_white_color_mask_h(self):
        """Filters out everything but white pixels"""

        light_thresh = self.pipeline.lane_filter.greyscale_threshold(self.l)

        # Use the threshold value just calculated and a separate saturation
        # threshold to filter out the lane pixels in the image. 

        lane_pixels = np.zeros_like(self.l)
        lightness_condition = self.l > light_thresh * .80
        lane_pixels[lightness_condition] = 1

        return lane_pixels

    def process_image(self):
        rate = rospy.Rate(10)

        print "Waiting for images..."
        while self.image is None:
            if rospy.is_shutdown():
                return
            rate.sleep()

        print "Waiting for activation..."
        while not rospy.is_shutdown():
            if (self.isActive):
                try:
                    # make copy of image
                    img = np.copy(self.image)

                    # resize image
                    img = self.resize_image(img, 1024)
                    #self.show_single_image("Initial Image", img)

                    # birdseye image
                    birdseye_img = self.pipeline.birdsEye.birdseye_view(img)
                    self.show_single_image("Birdseye", birdseye_img)

                    # white filter
                    filtered_binary_image = self.apply_h(birdseye_img)
                    #self.show_single_image("White Filter", filtered_binary_image * 255)

                    # adaptive thresholding
                    th2 = cv2.adaptiveThreshold(filtered_binary_image * 255,
                                                255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                cv2.THRESH_BINARY, 15, -2)
                    #self.show_single_image("Threshold Image", th2)

                    # grab specifications
                    horizontal = th2
                    rows, cols = horizontal.shape
                    horizontalsize = cols / 30
                    horizontalStructure = cv2.getStructuringElement(
                        cv2.MORPH_RECT, (horizontalsize, 2))

                    # mathematical morphological reconstruction
                    horizontal = cv2.erode(horizontal, horizontalStructure,
                                           (-3, -3))
                    horizontal = cv2.dilate(horizontal, horizontalStructure,
                                            (-1, -1))
                    #self.show_single_image("Reconstruction", horizontal)

                    # shape values
                    y, x = (horizontal > 0).nonzero()
                    height, width = horizontal.shape[:2]
                    max_y = np.amax(y)

                    # draw a line
                    cv2.line(horizontal, (0, max_y), (width, max_y),
                             (255, 0, 0), 3)
                    self.show_single_image("Final Image", horizontal)

                    # calculate distance
                    y_scale = 0.025  # tuned on a 2048x2048 image
                    deadband = 2.46
                    local_y = (height - max_y) * y_scale + deadband
                    print "Distance to Horizontal Line: " + str(local_y)

                    if local_y < 15 and local_y > deadband:
                        self.stop_line_distance.publish(local_y)
                    else:
                        self.stop_line_distance.publish(-1.0)

                    rate.sleep()
                except:
                    print(
                        "Horizontal Line Detection - No Horizontal Line Detected."
                    )
                    self.stop_line_distance.publish(-1.0)
                    pass


# Developer signal interrupt handler, immediate stop of ros_node_horizontal
def signalInterruptHandler(signum, frame):
    print("Horizontal Line Detection -", signum,
          "- Interrupting ROS Horizontal Pipeline...")
    sys.exit()


if __name__ == "__main__":
    # register CONTROL+C signal
    signal.signal(signal.SIGINT, signalInterruptHandler)

    # start pipeline
    node = LaneDetectionRosNodeHorizontal()
    node.process_image()
