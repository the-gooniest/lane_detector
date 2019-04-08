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
        self.lanes = []

        height = image_size[0]
        width = image_size[1]
        distort_offset = abs(road_points[1][0])
        self.birdseye_intercept = distort_offset * (float(width) / (
            distort_offset * 2 + width))
        self.birdseye_slope = self.birdseye_intercept / float(
            birdseye_points[1][1])

    def add_lane_width(self, windows):
        width = self.img.shape[1]
        if len(windows) == 2:
            if windows[1].found_lane() and windows[0].found_lane():
                diff = int(abs(windows[1].start_x() - windows[0].start_x()))
                if abs(diff - self.avg_lane_width) < width * 0.1:
                    if len(self.lane_widths) > 10:
                        self.lane_widths.pop(0)
                    self.lane_widths.append(diff)
                    self.avg_lane_width = int(np.mean(self.lane_widths))
                    # print "new lane width:", self.avg_lane_width

    def _find_hist_nonzero_region_centers(self, cropped_img, hist):
        """Returns the center pixels of all the non-zero regions in the given histogram"""

        cropped_width = cropped_img.shape[1]
        cropped_height = cropped_img.shape[0]

        # Get all nonzero indicies
        nonzero_indicies = [i for i, element in enumerate(hist) if element > 0]
        if len(nonzero_indicies) == 0:
            return []

        # Get centers of non-zero regions
        center_indicies = []
        if len(nonzero_indicies) == 1:
            center_indicies.append((nonzero_indicies[0], hist[nonzero_indicies[0]]))
        else:
            left_border_index = nonzero_indicies[0]
            for i, index in enumerate(nonzero_indicies[:-1]):
                other_index = nonzero_indicies[i + 1]
                if index + 1 != other_index:
                    center_index = (left_border_index + index) / 2
                    if left_border_index == index:
                        center_indicies.append((index, hist[index]))
                    else:
                        peak_value = np.sum(hist[left_border_index:index])
                        center_indicies.append((center_index, peak_value))
                    left_border_index = other_index

            # Make sure to append the last non-zero index
            if left_border_index < nonzero_indicies[-1]:
                peak_value = np.sum(hist[left_border_index:nonzero_indicies[-1]])
                center_indicies.append(
                    ((left_border_index + nonzero_indicies[-1]) / 2, peak_value))

        if len(center_indicies) == 0:
            return []

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

        # Remove the center indices with the smallest
        # max values until we have only two indices.
        i = 0
        while len(center_indicies) > 2:
            smallest_index = 0
            for i, center_index in enumerate(center_indicies):
                if center_index[1] < center_indicies[smallest_index][1]:
                    smallest_index = i
            center_indicies.pop(smallest_index)


        # Threshold indicies with sums that are too small 
        i = 0
        while i < len(center_indicies):
            if center_indicies[i][1] < 0.005 * cropped_width * cropped_height:
                center_indicies.pop(i)
            else:
                i += 1        

        # Unwrap max_val information
        for i in range(0, len(center_indicies)):
            if center_indicies[i]:
                center_indicies[i] = center_indicies[i][0]

        for center_index in center_indicies:
            if center_index is None:
                raise ValueError("center_index is None")

        return center_indicies

    def _lane_start_positions(self):
        """Returns the x pixel positions of the left and right lanes at the bottom of the image"""

        width = self.img.shape[1]
        height = self.img.shape[0]

        # Starts the summation 30% from the left side of the image
        # and ends at 30% from the right
        start_factor = 0.3
        start_pixel = int(width * start_factor)
        end_pixel = int(width * (1 - start_factor))

        peaks = []
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
                peaks[i] = peak + start_pixel

            if len(peaks) > 0:
                break
            k -= 1

        return peaks

    def extrapolate_start_position(self, start_position):
        """Calculates the start position of a lane using the known
        start position and the width of the other lane"""

        print "Extrapolating lane"

        width = self.img.shape[1]
        if start_position > width / 2:
            return start_position - self.avg_lane_width
        else:
            return start_position + self.avg_lane_width

    def assemble_lane_windows(self, start_positions):
        """Create and activate sliding windows"""

        width = self.img.shape[1]
        height = self.img.shape[0]

        for start_pos in start_positions:
            if start_pos < width * 0.3 or start_pos > width * 0.7:
               raise ValueError("Start positions outside of hist range")   

        for start_pos in start_positions:
            cv2.circle(self.birdseye_img, (start_pos, height - 3), 6, (0, 0, 255), -1)

        windows = []

        if len(start_positions) > 2:
            raise ValueError("Start positions length is greater than 2")

        for start_position in start_positions:
            windows.append(SlidingWindow(self, start_position))

        for window in windows:
            window.slide()

        if len(windows) == 1:
            window = windows[0]
            if window.found_lane():
                if abs(window.lane_heading()) < np.pi / 10:
                    start_pos = window.start_x()
                    new_start_pos = self.extrapolate_start_position(start_pos)
                    new_window = SlidingWindow(self, new_start_pos)
                    new_window.slide()
                    if new_window.found_lane():
                        windows.append(new_window)
                        windows.sort(key=lambda w: w.start_x())

        self.lanes = windows

    def _extract_lane_info(self):

        if len(self.lanes) == 0:
            print "No lanes detected"
            return

        # Get window with the most detected lane points
        largest_window = None
        largest_window_index = 0
        for i, window in enumerate(self.lanes):
            if window:
                if largest_window is None or len(window.x_lane_points) > len(largest_window.x_lane_points):
                    largest_window = window
                    largest_window_index = i

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
        center_offset = largest_window.start_x() - width / 2
        if len(self.lanes) == 2:
            print "Two lanes"
            if largest_window_index == 0:
                center_offset += self.avg_lane_width / 2
            else:
                center_offset -= self.avg_lane_width / 2
        else:
            print "One lane"
            # Left turn == left lane
            # Right turn == right lane
            if results['lane_heading'] < 0:
                center_offset += self.avg_lane_width / 2
            else:
                center_offset -= self.avg_lane_width / 2
        results['lane_center_offset'] = center_offset * WindowSlider.PIXELS_TO_METERS

        # Draw lane heading
        cv2.circle(self.birdseye_img, ((width / 2) + center_offset, height - 3), 6, (0, 255, 128), -1)
        start_line_point = ((width / 2) + center_offset, height)
        end_line_point = (int((width / 2) + center_offset - np.sin(results['lane_heading']) * 60), height - 60)
        cv2.line(self.birdseye_img, start_line_point, end_line_point, (0, 255, 128), 4)

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
        cv2.imshow("drawn", self.birdseye_img)
        cv2.waitKey(1)

        return results