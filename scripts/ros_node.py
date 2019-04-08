#!/usr/bin/env python2.7

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import os
import pickle
from pipeline import DetectionPipeline
import pprint
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray
import sys
import time
import traceback

class LaneDetectionRosNode:

    # Initialize class
    def __init__(self):
        rospy.init_node('lane_detector_node', anonymous=True)

        self.determineMode()

        self.sub = rospy.Subscriber(
            '/front_camera/image_raw', Image, self.image_callback, queue_size=1)
        self.lane_heading_pub = rospy.Publisher(
            '/lane_detection/lane_heading', Float32, queue_size=1)
        self.center_offset_pub = rospy.Publisher(
            '/lane_detection/lane_center_offset', Float32, queue_size=1)
        self.lane_width_pub = rospy.Publisher(
            '/lane_detection/lane_width', Float32, queue_size=1)
        self.lane_curvature_pub = rospy.Publisher(
            '/lane_detection/lane_curvature', Float32MultiArray, queue_size=1)
        self.left_lane_type_pub = rospy.Publisher(
            '/lane_detection/left_lane_type', Float32, queue_size=1)
        self.right_lane_type_pub = rospy.Publisher(
            '/lane_detection/right_lane_type', Float32, queue_size=1)

        # Initialize class members
        self.image = None
        calibration_dir = os.path.dirname(os.path.realpath(__file__))
        calibration_file = os.path.join(calibration_dir, "camera_calibration_data.p")
        self.calibration_data = pickle.load(open(calibration_file, "rb"))
        self.image_width = 768
        self.pipeline = DetectionPipeline((self.image_width, self.image_width), self.calibration_data, self.mode)

    # Determine which mode to execute
    def determineMode(self):
        if len(sys.argv) == 1:
            raise RuntimeError("Usage: rosrun lane_detection ros_node.py mode:=<lateral|obstacle|sign>")
            sys.exit(1)
        inputArg = sys.argv[1]
        print "Mode Selected: [" + str(inputArg) + "]"

        if(inputArg == "lateral"):
            self.mode = "lateral"
        elif(inputArg == "obstacle" or inputArg == "sign"):
            self.mode = "straight"
        else:
            raise RuntimeError("Usage: rosrun lane_detection ros_node.py mode:=<lateral|obstacle|sign>")
            sys.exit(1)

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

    # Resize image from ROS
    def resize_image(self, img, target_size):
        target_size = float(target_size)
        height = img.shape[0]
        width = img.shape[1]
        if height > width:
            scale_factor = target_size / height
            img = cv2.resize(
                img, (int(width * scale_factor), int(height * scale_factor)))
        else:
            scale_factor = target_size / width
            img = cv2.resize(
                img, (int(width * scale_factor), int(height * scale_factor)))
        return img

    # Process images through pipeline
    def process_images(self):
        rate = rospy.Rate(20)

        print "Waiting for images..."
        while self.image is None:
            if rospy.is_shutdown():
                return
            rate.sleep()

        print "Processing images..."
        while not rospy.is_shutdown():
            #start_time = time.time()

            # Run the pipeline on the newly collected image
            img_copy = np.copy(self.image)
            resized_img = self.resize_image(img_copy, self.image_width)

            #try:
            # mode "lateral" == results from lateral challenge
            # mode "straight" == results from stop sign and obstacle challenges
            results = self.pipeline.run(resized_img, self.mode)

            if results is not None:
                # Publish the lane heading
                if "lane_heading" in results:
                    lane_heading_msg = Float32()
                    lane_heading_msg.data = results['lane_heading']
                    self.lane_heading_pub.publish(lane_heading_msg)

                # Publish the lane center offset
                if "lane_center_offset" in results:
                    center_offset_msg = Float32()
                    center_offset_msg.data = results['lane_center_offset']
                    self.center_offset_pub.publish(center_offset_msg)

                # Publish the lane width
                if "lane_width" in results:
                    lane_width_msg = Float32()
                    lane_width_msg.data = results['lane_width']
                    self.lane_width_pub.publish(lane_width_msg)

                # Publish the lane width
                if "lane_curvature" in results:
                    lane_curvature_msg = Float32MultiArray()
                    lane_curvature_msg.data = results['lane_curvature']
                    self.lane_curvature_pub.publish(lane_curvature_msg)

                # Publish the left lane characterization
                if "left_lane_type" in results:
                    left_lane_type_msg = Float32()
                    left_lane_type_msg.data = results['left_lane_type']
                    self.left_lane_type_pub.publish(left_lane_type_msg)

                # Publish the right lane characterization
                if "right_lane_type" in results:
                    right_lane_type_msg = Float32()
                    right_lane_type_msg.data = results['right_lane_type']
                    self.right_lane_type_pub.publish(right_lane_type_msg)
            #except:
            #    print "Unexpected error occured in lane detection pipeline:"
            #    traceback.print_exc()
            #    sys.exit()

            #print "Execution time", time.time() - start_time
            rate.sleep()

if __name__ == "__main__":
    # Start ros node
    node = LaneDetectionRosNode()
    node.process_images()
