from birdseye import BirdsEye
import cv2
from helpers import show_single_image
from lanefilter import LaneFilter
from window_slider import WindowSlider as WSlider
from window_slider_lateral import WindowSlider as WSliderLateral
import numpy as np


class DetectionPipeline:

    def __init__(self, image_size, calibration_data, mode):
        self.image_size = image_size
        width, height = image_size
        self.camera_matrix = calibration_data["camera_matrix"]
        self.distortion_coefficients = calibration_data["distortion_coefficient"]

        # Setup undistortion matrices
        self.initialized_undistortion_maps = False
        self.initialize_undistortion_maps()

        # BirdsEye parameters
        top_crop_factor = 0.45
        crop_height_factor = 0.2
        distort_factor = 10 * crop_height_factor
        elgonation_factor = 3.0

        # Initialize the "before distortion" points
        cropped_height = int(crop_height_factor * height * elgonation_factor)
        self.birdseye_points = [
            (0, 0),  # Top Left
            (0, cropped_height),  # Bottom left
            (width, cropped_height),  # Bottom right
            (width, 0)  # Top right
        ]

        # Initialize the "after distortion" points
        top_crop = int(height * top_crop_factor)
        bottom_crop = int((top_crop_factor + crop_height_factor) * height)
        distort_left = int(-width * distort_factor)
        distort_right = int(width * (1 + distort_factor))
        self.road_points = [
            (0, top_crop),  # Top Left
            (distort_left, bottom_crop),  # Bottom left
            (distort_right, bottom_crop),  # Bottom right
            (width, top_crop)  # Top right
        ]

        # Initialize BirdsEye instance
        self.birdsEye = BirdsEye(self.road_points, self.birdseye_points)

        # Initialize LaneFilter instance
        self.lane_filter = LaneFilter()

        # Setup WindowSlider instance
        if mode == "straight":
            self.swindow = WSlider(image_size, self.birdseye_points, self.road_points)
        elif mode == "lateral":
            self.swindow = WSliderLateral(image_size, self.birdseye_points, self.road_points)
        else:
            raise RuntimeError("Invalid Mode Provided.")

    def initialize_undistortion_maps(self):
        """Initializes the matrix used to fix camera spherical distortion"""

        new_camera_matrix, valid_roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.distortion_coefficients, self.image_size,
            0)

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.distortion_coefficients, None,
            new_camera_matrix, self.image_size, cv2.CV_16SC2)

    def get_undistorted_image(self, image):
        """Calculates undistorted image"""

        undistorted_image = cv2.remap(image, self.map1, self.map2,
                                      cv2.INTER_LINEAR)

        return undistorted_image

    def draw_birdseye_points(self, image, points):
        """Draws self.road_points and self.birdseye_points"""
        point_image = np.copy(image)
        for point in points:
            cv2.circle(point_image, point, 5, [0, 0, 255], cv2.FILLED)

        cv2.imshow('image1', point_image)
        cv2.waitKey(1)

    def run(self, image, mode):
        """Returns detected lane information"""

        # Undistort image
        #undistorted_image = self.get_undistorted_image(image)

        # Bird's eye
        birdseye_img = self.birdsEye.birdseye_view(image)

        # Lane Filter
        filtered_binary_image = self.lane_filter.apply(birdseye_img)
        self.filtered_binary_image = filtered_binary_image * 255
        # cv2.imshow('filtered', self.filtered_binary_image)

        # Gather characteristics from the lanes found
        results = self.swindow.find_lanes(filtered_binary_image, birdseye_img)

        # return results
