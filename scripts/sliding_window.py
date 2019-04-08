import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
warnings.simplefilter('ignore', np.RankWarning)

class SlidingWindow:
    SOLID = 0
    DASHED = 1
    NONE = -1
    LEFT = 0
    RIGHT = 1
    MINIMUM_LANE_POINTS = 3

    def __init__(self, window_slider, start_position, start_angle=0):
        self.window_slider = window_slider

        width = self.window_slider.img.shape[1]
        height = self.window_slider.img.shape[0]

        size_factor = 0.07
        if height < width:
            self.length = int(height * size_factor)
        else:
            self.length = int(width * size_factor)

        self.center = (start_position, height - self.length / 2)
        self.last_slope_angle = start_angle
        self.x_lane_points = []
        self.y_lane_points = []
        self.bad_x_lane_points = []
        self.bad_y_lane_points = []
        self.good_lane = True
        self.box_translation_dist = self.length / 2
        self.lane_type = SlidingWindow.NONE

    def angle_diff(self, a1, a2):
        """Calculates the difference between two angles"""
        return np.pi - abs(abs(a1 - a2) - np.pi)

    def _get_box_center_offset(self, slope, intercept):
        """Returns the offset of the box's center from the mid-point of the line intersecting the given box"""
        intercept = intercept
        points_hash = {}

        if slope == 0:
            return (0, int(intercept - self.length / 2))

        edge_points = [(0, intercept), (self.length, slope * self.length + intercept),
                       (-intercept / slope, 0), ((self.length - intercept) / slope, self.length)]

        for edge_point in edge_points:
            if edge_point[0] >= 0 and edge_point[0] <= self.length:
                if edge_point[1] >= 0 and edge_point[1] <= self.length:
                    points_hash[edge_point[1]] = edge_point[0]

        if len(points_hash) > 2:
            raise ValueError('Too many points added to the points hash')

        x_center = int(np.sum([x for x in points_hash]) / 2)
        y_center = int(np.sum([points_hash[x] for x in points_hash]) / 2)
        return (x_center - self.length / 2, y_center - self.length / 2)

    def _predict_and_set_position_of_next_box(self, slope_angle):
        # Extrapolate next box position using current slope
        # Check if the slope is in the right quadrant.
        # Assumption: new box's slope should be close to last box's slope
        # (within pi/2 radians)
        original_angle = slope_angle
        #offset = np.pi / 10000
        #slope_angle = np.clip(slope_angle, self.last_slope_angle - offset,  self.last_slope_angle + offset)
        # slope_angle = np.clip(slope_angle, -np.pi / 3, np.pi / 3)
        #print abs(slope_angle - original_angle)

        #diff = self.angle_diff(slope_angle, self.last_slope_angle)
        #if diff > np.pi / 2:
        #    slope_angle += np.pi
        self.last_slope_angle = slope_angle

        # Set center coordinates for next box
        x_center = int(self.center[0] - np.sin(slope_angle) * self.box_translation_dist)
        y_center = int(self.center[1] - np.cos(slope_angle) * self.box_translation_dist)
        self.center = (x_center, y_center)

    def draw_lane(self):
        # Draw new center point and box dimensions
        if not self.found_lane():
            return

        if self.lane_type == SlidingWindow.SOLID:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        for i, x in enumerate(self.x_lane_points[:-1]):
            point1 = (x, self.y_lane_points[i])
            point2 = (self.x_lane_points[i + 1], self.y_lane_points[i + 1])
            cv2.line(self.window_slider.birdseye_img, point1, point2, color, 2)

    def draw_boxes(self):
        if not self.found_lane():
            return

        for i, x in enumerate(self.x_lane_points):
            point = (x, self.y_lane_points[i])
            cv2.rectangle(
                self.window_slider.birdseye_img,
                (point[0] - self.length / 2, point[1] - self.length / 2),
                (point[0] + self.length / 2, point[1] + self.length / 2),
                (0, 0, 255), 2)
        """
        for i, x in enumerate(self.bad_x_lane_points):
            point = (x, self.bad_y_lane_points[i])
            cv2.rectangle(
                self.window_slider.birdseye_img,
                (point[0] - self.length / 2, point[1] - self.length / 2),
                (point[0] + self.length / 2, point[1] + self.length / 2),
                (255, 255, 0), 2)
        """

    def found_lane(self):
        return len(self.x_lane_points) >= SlidingWindow.MINIMUM_LANE_POINTS

    def _center_inside_image(self):
        """Returns true if the sliding window's center is still inside the image"""

        width = self.window_slider.img.shape[1]
        height = self.window_slider.img.shape[0]

        ((left, top), (right, bottom)) = self._box_bounds()
        if top <= 0 or left <= 0 or right >= width:
            return False

        slope = self.window_slider.birdseye_slope
        intercept = self.window_slider.birdseye_intercept

        offset = 5
        dist_from_bottom = height - self.center[1]
        left_bound = intercept - (slope * dist_from_bottom) + offset
        if self.center[0] < left_bound:
            return False

        right_bound = (width - intercept) + (slope * dist_from_bottom) - offset
        if self.center[0] > right_bound:
            return False
        return True

    def _box_bounds(self):
        # Calculate box dimensions
        return (
            (self.center[0] - self.length / 2,
             self.center[1] - self.length / 2),  # left, top
            (self.center[0] + self.length / 2,
             self.center[1] + self.length / 2))  # right, bottom

    def lane_side(self):
        """Returns which side, left or right, the lane is on relative to the car's center"""
        if not self.found_lane():
            return None

        x_start = self.start_x()
        width = self.window_slider.img.shape[1]
        if x_start < width / 2:
            return SlidingWindow.LEFT
        else:
            return SlidingWindow.RIGHT

    def start_x(self):
        if not self.found_lane():
            return None
        return self.x_lane_points[0]

    def lane_heading(self):
        """Returns the angle of the lane relative to the car"""
        if not self.found_lane():
            return None

        x_points = self.x_lane_points[:SlidingWindow.MINIMUM_LANE_POINTS]
        y_points = self.y_lane_points[:SlidingWindow.MINIMUM_LANE_POINTS]
        slope, intercept = np.polyfit(y_points, x_points, 1)
        return np.arctan(slope)

    def slide(self):
        """Finds lane points by searching pixels within a sliding window"""

        width = self.window_slider.img.shape[1]
        height = self.window_slider.img.shape[0]

        window_count = 0
        while True:
            window_count += 1

            # Make sure the box is still located inside the image
            if not self._center_inside_image():
                break

            # Check if the box is empty
            ((left, top), (right, bottom)) = self._box_bounds()
            box_image = self.window_slider.img[top:bottom, left:right]
            hist_sum = np.sum(box_image)
            if hist_sum < self.length * self.length * 0.02:
                self.bad_x_lane_points.append(self.center[0])
                self.bad_y_lane_points.append(self.center[1])
                self._predict_and_set_position_of_next_box(self.last_slope_angle)
                continue

            # Polyfit the white pixels within the box
            non_zero_indicies = np.nonzero(box_image)
            slope = self.last_slope_angle
            try:
                # Reverse x and y for the polyfit because lanes are more likely to proceed
                # vertically down the image instead of horizontally
                slope, intercept = np.polyfit(non_zero_indicies[0], non_zero_indicies[1], 1)
            except ValueError as err:
                # Polyfit is finiky, just ignore its value errors
                pass

            # Move box to the center of detected the lane pixels
            box_center_offset = self._get_box_center_offset(slope, intercept)
            self.center = (self.center[0] + box_center_offset[0],
                           self.center[1] + box_center_offset[1])

            # Record detected lane point
            self.x_lane_points.append(self.center[0])
            self.y_lane_points.append(self.center[1])

            self._predict_and_set_position_of_next_box(np.arctan(slope))

        if not self.found_lane():
            return

        # Add a start point to the list of points
        ml_points = SlidingWindow.MINIMUM_LANE_POINTS
        slope, intercept = np.polyfit(self.y_lane_points[:ml_points],
                                      self.x_lane_points[:ml_points], 1)
        self.x_lane_points.insert(0, int(slope * height + intercept))
        self.y_lane_points.insert(0, height)

        # Classify lane
        window_points_ratio = float(len(self.x_lane_points)) / window_count
        if window_points_ratio < 0.5:
            self.lane_type = SlidingWindow.DASHED
        else:
            self.lane_type = SlidingWindow.SOLID

        self.draw_lane()
        self.draw_boxes()
