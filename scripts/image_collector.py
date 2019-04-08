#!/usr/bin/env python2.7

import time
import sys
import cv2
from helpers import show_single_image
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageCollector:

    def __init__(self):
        rospy.init_node('image_collector', anonymous=True)
        self.bridge = CvBridge()
        self.sub = rospy.Subscriber('/front_camera/image_raw', Image,
                                    self.image_callback)
        self.image = None
        self.image_folder = './collected_images'

    def image_callback():
        # check if image data exists
        if image_msg.data is None:
            raise RuntimeError("Null image data read from message.")

        # convert to cv2 mat type
        image = bridge.imgmsg_to_cv2(image_msg, "bgr8")

        # if image is null, error
        if image is None:
            raise RuntimeError("Decoded image is null")

        self.image = image

    def new_image_file_name(self):
        return self.image_folder + str(time.time()) + '.jpg'

    def collect(self):
        rate = rospy.Rate(15)

        # wait for the subscriber to preload an image
        print 'Waiting for image...'
        while self.image is None and not rospy.is_shutdown():
            rate.sleep()

        while not rospy.is_shutdown():
            if self.image is not None:
                file_name = new_image_file_name()
                print 'saving ' + file_name
                cv2.imwrite(file_name, self.image)
            rate.sleep()


if __name__ == "__main__":
    image_collector = ImageCollector()
    image_collector.collect()
