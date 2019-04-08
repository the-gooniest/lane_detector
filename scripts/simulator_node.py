#!/usr/bin/env python

import argparse
import base64
import cv2
from helpers import show_single_image
import numpy as np
import rospy
import pickle
from pipeline import DetectionPipeline
from std_msgs.msg import String


class LaneDetectionNode:

    def __init__(self, calibration_image):
        self.mode = 'straight'
        self.image_width = 768
        self.calibration_image = calibration_image
        self.parse_arguments()
        self.calibration_data = pickle.load(
            open("camera_calibration_data.p", "rb"))
        self.pipeline = DetectionPipeline((self.image_width, self.image_width), self.calibration_data, self.mode)
        self.create_publisher()
        self.create_subscriber()

    def parse_arguments(self):
        parser = argparse.ArgumentParser(
            description='Subscriber node for lane detection algorithm.')
        parser.add_argument(
            '-v', '--verbose', dest='verbose', action='store_true')
        parser.add_argument(
            '-i', '--show-images', dest='show_images', action='store_true')
        parser.set_defaults(verbose=False)
        self.args = parser.parse_args()

    def create_subscriber(self):
        """Creates a subscriber node that reads and processes image messages"""
        print 'initializing lane detection subscriber'
        rospy.init_node('lane_detector_node', anonymous=True)
        sub = rospy.Subscriber('/autodrive_sim/output/camera_front', String, self.callback)

    def create_publisher(self):
        """Creates a publisher node that outputs the processed lane-detected image"""
        print 'initializing lane detection publisher'
        self.pub = rospy.Publisher(
            '/autodrive_sim/input/image_in', String, queue_size=1)

    def resize_image(self, img, target_size):
        """Resize image from ROS"""
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

    def callback(self, image_msg):
        print "Received message!"
        if image_msg.data is None:
            raise RuntimeError("Null image data read from message")

        # Convert image_msg.data from base64-string to OpenCV mat format
        decoded_string = image_msg.data.decode('base64')
        np_array = np.fromstring(decoded_string, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError("Decoded image is null")
        if image.shape[0] != self.image_width:
            return

        # Run the pipeline on the newly collected image
        # resized_img = self.resize_image(image, self.image_width)
        # cv2.imshow('filtered', image)
        # cv2.waitKey(1)

        #try:
        # mode "lateral" == results from lateral challenge
        # mode "straight" == results from stop sign and obstacle challenges
        results = self.pipeline.run(image, self.mode)

        # Process image in lane_detector module
        output_msg = String()
        retval, jpeg_byte_array = cv2.imencode(
            '.jpg', self.pipeline.swindow.birdseye_img)
        np_string = jpeg_byte_array.tostring()
        encoded_string = np_string.encode('base64')
        output_msg.data = encoded_string

        # Send message
        self.pub.publish(output_msg)

        if self.args.show_images:
            cv2.imshow("image", self.pipeline.result_projected_image)
            cv2.waitKey(1)
        


if __name__ == "__main__":
    test_image = cv2.imread("test_image.jpg")
    LaneDetectionNode(test_image)
    rospy.spin()
