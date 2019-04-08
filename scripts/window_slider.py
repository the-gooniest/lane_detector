import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from sliding_window import SlidingWindow
import time
import warnings
warnings.simplefilter('ignore', np.RankWarning)

class WindowSlider:
    PIXELS_TO_METERS = 0.92 / 16.5

    def __init__(self, image_size, birdseye_points, road_points):
        self.avg_lane_width = 71
        self.lane_widths = [71]
        self.lanes = [None, None]

        height = image_size[0]
        width = image_size[1]
        distort_offset = abs(road_points[1][0])
        self.birdseye_intercept = distort_offset * (float(width) / (
            distort_offset * 2 + width))
        self.birdseye_slope = self.birdseye_intercept / float(
            birdseye_points[1][1])

    def add_lane_width(self, windows):
        width = self.img.shape[1]
        if windows[0] and windows[1]:
            if windows[1].found_lane() and windows[0].found_lane():
                diff = int(abs(windows[1].start_x() - windows[0].start_x()))
                if abs(diff - self.avg_lane_width) < width * 0.1:
                    if len(self.lane_widths) > 10:
                        self.lane_widths.pop(0)
                    self.lane_widths.append(diff)
                    self.avg_lane_width = int(np.mean(self.lane_widths))

    def _find_hist_nonzero_region_centers(self, cropped_img, hist):
        """Returns the center pixels of all the non-zero regions in the given histogram"""

        # Get all nonzero indicies
        nonzero_indicies = [i for i, element in enumerate(hist) if element > 0]
        if len(nonzero_indicies) == 0:
            return [None, None]

        # Get centers of non-zero regions
        center_indicies = []
        if len(nonzero_indicies) == 1:
            center_indicies.append((nonzero_indicies[0],
                                    hist[nonzero_indicies[0]]))
        else:
            left_border_index = nonzero_indicies[0]
            for i, index in enumerate(nonzero_indicies[:-1]):
                other_index = nonzero_indicies[i + 1]
                if index + 1 != other_index:
                    center_index = (left_border_index + index) / 2
                    if left_border_index == index:
                        center_indicies.append((index, hist[index]))
                    else:
                        peak_value = np.max(hist[left_border_index:index])
                        center_indicies.append((center_index, peak_value))
                    left_border_index = other_index

            # Make sure to append the last non-zero index
            if left_border_index < nonzero_indicies[-1]:
                peak_value = np.max(
                    hist[left_border_index:nonzero_indicies[-1]])
                center_indicies.append(
                    ((left_border_index + nonzero_indicies[-1]) / 2,
                     peak_value))

        if len(center_indicies) == 0:
            return [None, None]

        # Make sure that the starting positions are appropriately spaced apart
        if len(center_indicies) >= 2:
            i = 0
            while i < len(center_indicies) - 1:
                index1 = center_indicies[i]
                index2 = center_indicies[i + 1]
                if index2[0] - index1[0] < self.avg_lane_width * 0.7:
                    if index1[1] > index2[1]:
                        center_indicies.pop(i + 1)
                    else:
                        center_indicies.pop(i)
                else:
                    i += 1

        start_indices = [None, None]

        # Find largest left lane index
        for center in center_indicies:
            if center[0] < len(hist) / 2:
                if start_indices[0] is None or start_indices[0][1] < center[1]:
                    start_indices[0] = center

        # Find largest right lane index
        for center in center_indicies:
            if center[0] >= len(hist) / 2:
                if start_indices[1] is None or start_indices[1][1] < center[1]:
                    start_indices[1] = center

        # Unwrap max_val information
        for i in range(0, len(start_indices)):
            if start_indices[i]:
                start_indices[i] = start_indices[i][0]

        return start_indices

    def _lane_start_positions(self):
        """Returns the x pixel positions of the left and right lanes at the bottom of the image"""

        width = self.img.shape[1]
        height = self.img.shape[0]

        # Starts the summation 30% from the left side of the image
        # and ends at 30% from the right
        start_factor = 0.3
        start_pixel = int(width * start_factor)
        end_pixel = int(width * (1 - start_factor))

        peaks = [None, None]
        crop_factor = 6
        k = crop_factor - 1
        while k > 0:
            # Sum all of the white pixels in each column of the image
            top_crop = (height * k) / crop_factor
            cropped_img = self.img[top_crop:, start_pixel:end_pixel]
            hist = np.sum(cropped_img, axis=0)

            # Find peaks in histogram
            peaks = self._find_hist_nonzero_region_centers(cropped_img, hist)

            # Add start_pixel offset to peaks
            for i, peak in enumerate(peaks):
                if peak:
                    peaks[i] = peak + start_pixel

            if peaks[0] or peaks[1]:
                break
            k -= 1
        return peaks

    def assemble_lane_windows(self, start_positions):
        """Create and activate sliding windows"""

        width = self.img.shape[1]
        height = self.img.shape[0]

        windows = [None, None]
        start_angle = [-np.pi, -np.pi]

        if len(start_positions) < 2:
            raise ValueError("Start positions length is less than 2")

        for i in range(0, 2):
            other_lane = (i + 1) % 2
            if start_positions[i] is None:
                # Check if we can extrapolate this lane's position from the other lane's position
                if start_positions[other_lane] is not None:
                    lane_width = -self.avg_lane_width if i == 0 else self.avg_lane_width
                    start_positions[i] = start_positions[other_lane] + lane_width
                else:
                    # Use previous lane data to check for this lane
                    if self.lanes[i] and self.lanes[i].found_lane():
                        start_positions[i] = self.lanes[i].start_x()
                        start_angle[i] = -self.lanes[i].lane_heading()

            if start_positions[i]:
                windows[i] = SlidingWindow(self, start_positions[i], start_angle[i])

        for window in windows:
            if window:
                window.slide()

        # Check for windows moving to the other side during lane changes
        if windows[0] and windows[0].found_lane() and windows[0].start_x() >= width / 2:
            windows[1] = windows[0]
            windows[0] = None
        elif windows[1] and windows[1].found_lane() and windows[1].start_x() < width / 2:
            windows[0] = windows[1]
            windows[1] = None

        self.lanes = windows

    def _extract_lane_info(self):

        # Get window with the most detected lane points
        largest_window = None
        for i, window in enumerate(self.lanes):
            if window:
                if largest_window is None or len(window.x_lane_points) > len(
                        largest_window.x_lane_points):
                    largest_window = window

        if largest_window is None or not largest_window.found_lane():
            print "No lanes Detected\n"
            return None

        height = self.img.shape[0]
        width = self.img.shape[1]

        # Record lane width and heading
        results = {}
        results['lane_width'] = self.avg_lane_width * WindowSlider.PIXELS_TO_METERS
        results['lane_heading'] = largest_window.lane_heading()

        # Record lane center offset
        results['lane_center_offset'] = width / 2 - largest_window.start_x()
        if largest_window.lane_side() == SlidingWindow.LEFT:
            results['lane_center_offset'] -= self.avg_lane_width / 2
        else:
            results['lane_center_offset'] += self.avg_lane_width / 2
        results['lane_center_offset'] *= WindowSlider.PIXELS_TO_METERS

        # Record lane types
        if self.lanes[0]:
            results['left_lane_type'] = self.lanes[0].lane_type
        else:
            results['left_lane_type'] = -1

        if self.lanes[1]:
            results['right_lane_type'] = self.lanes[1].lane_type
        else:
            results['right_lane_type'] = -1

        # Get 3rd degree polynomial best fit
        poly_degree = 3
        inverted_y_points = []
        for i in range(0, len(largest_window.y_lane_points)):
            inverted_y_points.append(height - largest_window.y_lane_points[i])

        coefficients = np.polyfit(inverted_y_points, largest_window.x_lane_points, poly_degree)

        # Fit center line
        derivative_coef = np.polyder(coefficients)
        lane_offset = self.avg_lane_width / 2

        # Sample points in the center of the lane
        x_center_line = []
        y_center_line = []
        i = height
        while i > 0:
            tangent = np.polyval(derivative_coef, i)

            slope_angle = -np.arctan(tangent)
            if largest_window.lane_side() == SlidingWindow.RIGHT:
                slope_angle -= np.pi / 2
            else:
                slope_angle += np.pi / 2

            x_val = int(np.polyval(coefficients, i))
            # cv2.circle(self.birdseye_img, (x_val, height - i), 3, (255, 0, 0), -1)

            x_val = int(x_val + np.sin(slope_angle) * lane_offset)
            y_val = int(i - np.cos(slope_angle) * lane_offset)

            x_center_line.append(x_val)
            y_center_line.append(y_val)

            i -= 10

        # Polyfit center points and record lane curvature polynomial
        center_coef = np.polyfit(y_center_line, x_center_line, poly_degree)

        # Draw center curve
        i = height / 2
        while i > 0:
            x_val = int(np.polyval(center_coef, i))
            cv2.circle(self.birdseye_img, (x_val, height - i), 3, (0, 255, 255), -1)
            i -= 10

        # Scale coefficients to the world space
        intercept_offset = np.polyval(center_coef, 0)
        center_coef[-1] -= intercept_offset
        #for i in range(0, len(center_coef)):
        #    center_coef[i] /= WindowSlider.PIXELS_TO_METERS

        results['lane_curvature'] = center_coef

        return results

    def find_lanes(self, img, birdseye_img):
        self.img = img
        self.birdseye_img = np.copy(birdseye_img)

        # Get line edge of birdseye image
        height = self.img.shape[0]
        width = self.img.shape[1]

        # Find where the lanes start at the bottom of the image
        start_positions = self._lane_start_positions()

        # Create and slide windows
        self.assemble_lane_windows(start_positions)

        # Calculate lane width from newly discovered lanes
        self.add_lane_width(self.lanes)
        
        # Extract lane info from the lanes
        results = self._extract_lane_info()

        # Draw the lanes
        # cv2.imshow("drawn", self.birdseye_img)
        # cv2.waitKey(1)

        return results