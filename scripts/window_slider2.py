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
        initial_lane_width = 80
        self.avg_lane_width = initial_lane_width
        self.lane_widths = [initial_lane_width]
        self.lanes = [None, None]

        height = image_size[0]
        width = image_size[1]
        distort_offset = abs(road_points[1][0])
        self.birdseye_intercept = distort_offset * (float(width) / (
            distort_offset * 2 + width))
        self.birdseye_slope = self.birdseye_intercept / float(
            birdseye_points[1][1])

    def has_prev_lane_data(self):
        return self.lanes[0] is not None and self.lanes[1] is not None

    def add_lane_width(self, windows):
        width = self.img.shape[1]
        if windows[0] and windows[1]:
            if windows[0].found_lane() and windows[1].found_lane():
                diff = int(abs(windows[1].start_x() - windows[0].start_x()))
                diff = np.clip(diff, diff / 2, (diff * 3) / 2)
                if abs(diff - self.avg_lane_width) < width * 0.1:
                    if len(self.lane_widths) > 10:
                        self.lane_widths.pop(0)
                    self.lane_widths.append(diff)
                    self.avg_lane_width = int(np.mean(self.lane_widths))

    def _find_hist_nonzero_region_centers(self):
        """Returns the center pixels of all the non-zero regions in the given histogram"""

        width = self.img.shape[1]
        height = self.img.shape[0]

        # Starts the summation 30% from the left side of the image
        # and ends at 30% from the right
        start_factor = 0.3
        start_pixel = int(width * start_factor)
        end_pixel = int(width * (1 - start_factor))

        # Find the potential lane start positions
        crop_factor = 6
        k = crop_factor - 1
        center_indicies = []
        while k > 0:
            # Sum all of the white pixels in each column of the cropped image
            top_crop = (height * k) / crop_factor
            cropped_img = self.img[top_crop:, start_pixel:end_pixel]
            hist = np.sum(cropped_img, axis=0)

            # Get all nonzero indicies
            nonzero_indicies = [i for i, element in enumerate(hist) if element > 0]
            if len(nonzero_indicies) == 0:
                k -= 1
                continue

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
                            peak_value = np.max(hist[left_border_index:index])
                            center_indicies.append((center_index, peak_value))
                        left_border_index = other_index

                # Make sure to append the last non-zero index
                if left_border_index < nonzero_indicies[-1]:
                    peak_value = np.max(hist[left_border_index:nonzero_indicies[-1]])
                    center_indicies.append(((left_border_index + nonzero_indicies[-1]) / 2, peak_value))

            # If we found any center_indicies, we're good to go
            if len(center_indicies) > 0:
                break        
            k -= 1

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
                    
        # Unwrap max_val information
        for i in range(0, len(center_indicies)):
            center_indicies[i] = center_indicies[i][0]

        # Add start_pixel offset to peaks
        for i in range(0, len(center_indicies)):
            center_indicies[i] = center_indicies[i] + start_pixel

        return center_indicies

    def extrapolate_other_start_position(self, start_positions):
        """Calculates the start position of a lane with an unknown start position
        using a known starting position and the width of the lane"""

        if start_positions[0] is None and start_positions[1] is None:
            return start_positions

        if start_positions[0] is not None and start_positions[1] is not None:
            return start_positions

        print "Extrapolating lane"

        good_lane_index = 0
        if start_positions[1] is not None:
            good_lane_index = 1
        bad_index = (good_lane_index + 1) % 2

        img_center = self.img.shape[1] / 2
        lane_width = (bad_index - good_lane_index) * self.avg_lane_width
        start_positions[bad_index] = img_center + lane_width

    def check_for_lane_change(self, center_indicies, start_positions):

        if not self.has_prev_lane_data():
            return False

        width = self.img.shape[1]

        # Match the lane centers to the previous lane centers
            # Find previous lane that most closely matches the center most point of the center indices
            # If that point has crossed the center axis (ex: left lane now on the right side of the image...)
                # check for an available center index on the opposite side of the lane
                # if the center point exists, a lane change occured
                # else, keep the same left/right lane classifications

        # Find matches
        matches = [None, None]
        for i in range(0, 2):
            if self.lanes[i]:
                start_x = self.lanes[i].start_x()
                matches[i] = 0
                for j in range(1, len(center_indicies)):
                    if abs(center_indicies[j] - start_x) < abs(center_indicies[matches[i]] - start_x):
                        matches[i] = j

        # Get closest match
        closest_match = None
        for i in range(0, 2):
            if self.lanes[i]:
                other_index = (i+1)%2
                if closest_match is None:
                    closest_match = i
                elif not self.lanes[other_index]:
                    closest_match = i
                else:
                    dist1 = abs(center_indicies[matches[i]] - self.lanes[i].start_x())
                    dist2 = abs(center_indicies[matches[other_index]] - self.lanes[other_index].start_x())
                    if dist1 < dist2:
                        closest_match = i

        match_index = center_indicies[matches[closest_match]]

        # Check if match is close enough
        if abs(match_index - self.lanes[closest_match].start_x()) < 20:
            if closest_match is 1 and match_index - width / 2 < 0:
                # If there is another lane to the right of the previously known right lane
                if matches[closest_match] < len(center_indicies) - 1:
                    start_positions[0] = center_indicies[matches[closest_match]]
                    start_positions[1] = center_indicies[matches[closest_match] + 1]
                    print "found right lane change"
                    return True
                else:
                    print "No right lane to change to, holding lane"
                    start_positions[0] = center_indicies[matches[closest_match] - 1]
                    start_positions[1] = center_indicies[matches[closest_match]]
                    return True

            # Check for right lane change
            elif closest_match is 0 and match_index - width / 2 > 0:
                # If there is another lane to the right of the previously known right lane
                if matches[closest_match] > 0:
                    start_positions[0] = center_indicies[matches[closest_match] - 1]
                    start_positions[1] = center_indicies[matches[closest_match]]
                    print "found left lane change"
                    return True
                else:
                    print "No left lane to change to, holding lane"
                    start_positions[0] = center_indicies[matches[closest_match]]
                    start_positions[1] = center_indicies[matches[closest_match] + 1]
                    return True

        return False

    def _lane_start_positions(self):
        """Returns the x pixel positions of the left and right lanes at the bottom of the image"""

        width = self.img.shape[1]
        height = self.img.shape[0]
        case = 0
        start_positions = [None, None]
        center_indicies = self._find_hist_nonzero_region_centers()

        # Case 1: No center indices
            # Use previous lane's center indices
            # Else, if no previous lane data
                # return none for either lane
        if len(center_indicies) == 0:
            print "CASE 1"
            case = 1
            for i in range(0, 2):
                if self.lanes[i]:
                    start_positions[i] = self.lanes[i].start_x()

        # Case 2: One center index
            # If previous lane data exists
                # Find which lane side is closer to the index
                # Use the center index as the start position for the closer lane side
                # Extrapolate other lane start position using the lane width
            # Else,
                # Check which side of the image the index is on
                # Set the corresponding lane to that index
                # Extrapolate the other lane
        elif len(center_indicies) == 1:
            print "CASE 2"
            case = 2
            center_index = center_indicies[0]
            if self.has_prev_lane_data():
                print "Has lane data"
                closer_lane = None
                for i in range(0, 2):
                    if self.lanes[i]:
                        if closer_lane is None:
                            closer_lane = i
                        else:
                            if abs(self.lanes[i].start_x() - center_index) < abs(self.lanes[closer_lane].start_x() - center_index):
                                closer_lane = i
                start_positions[closer_lane] = center_index
            else:
                print "Does not have lane data"
                if center_index < width / 2:
                    start_positions[0] = center_index
                else:
                    start_positions[1] = center_index
            self.extrapolate_other_start_position(start_positions)

        # Case 3: More than one center index
            # Check for a lane change
            # Else,
                # Pick the two most center lanes
        else:
            case = 3
            print "CASE 3"
            if not self.check_for_lane_change(center_indicies, start_positions):
                print "No lane change check"
                most_center = center_indicies[0]
                sec_most_center = None
                for center in center_indicies[1:]:
                    if abs(center - width / 2) < abs(most_center - width / 2):
                        sec_most_center = most_center
                        most_center = center
                    elif sec_most_center is None or abs(center - width / 2) < abs(sec_most_center - width / 2):
                        sec_most_center = center

                if most_center < sec_most_center:
                    start_positions[0] = most_center
                    start_positions[1] = sec_most_center
                else:
                    start_positions[0] = sec_most_center
                    start_positions[1] = most_center
                print start_positions

        # Finally, return the start positions
        return start_positions, case

    def assemble_lane_windows(self, start_positions):
        """Create and activate sliding windows"""

        width = self.img.shape[1]
        height = self.img.shape[0]

        windows = [None, None]

        if start_positions[0] is None or start_positions[1] is None:
            raise ValueError("Start positions have null value before assembling sliding windows")

        # draw start positions
        cv2.circle(self.birdseye_img, (start_positions[0], height - 3), 6, (0, 255, 0), -1)
        cv2.circle(self.birdseye_img, (start_positions[1], height - 3), 6, (0, 0, 255), -1)

        # Try to get a better estimate for the direction the window
        # should initially start sliding using past lane data.
        start_angle = None
        if self.has_prev_lane_data():
            headings = []
            for i in range(0, 2):
                if self.lanes[i]:
                    headings.append(self.lanes[i].lane_heading())
            #print np.average(headings), -np.pi
            start_angle = np.average(headings)      

        for i in range(0, 2):
            if start_angle:
                windows[i] = SlidingWindow(self, start_positions[i], start_angle)
            else:
                windows[i] = SlidingWindow(self, start_positions[i])

        # Slide the windows
        for window in windows:
            window.slide()

        # Remove windows that didn't find a lane
        if windows[0] and not windows[0].found_lane():
            windows[0] = None
        elif windows[1] and not windows[1].found_lane():
            windows[1] = None

        # Record past lane data
        self.lanes = windows
   
    def _extract_lane_info(self):

        # Get SlidingWindow with the most detected lane points
        largest_window = None
        for i, window in enumerate(self.lanes):
            if window:
                if largest_window is None or len(window.x_lane_points) > len(
                        largest_window.x_lane_points):
                    largest_window = window

        if largest_window is None or not largest_window.found_lane():
            #print "No lanes Detected\n"
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

        # draw the lane heading
        

        # Get 3rd degree polynomial best fit
        poly_degree = 3
        inverted_y_points = []
        for i in range(0, len(largest_window.y_lane_points)):
            inverted_y_points.append(height - largest_window.y_lane_points[i])

        coefficients = np.polyfit(inverted_y_points,
                                  largest_window.x_lane_points, poly_degree)

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
            #cv2.circle(self.birdseye_img, (x_val, height - i), 3, (0, 255, 255), -1)
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

        # initialize results dictionary
        results = None

        # Get line edge of birdseye image
        height = self.img.shape[0]
        width = self.img.shape[1]

        # Find where the lanes start at the bottom of the image
        start_positions, case = self._lane_start_positions()

        # Stop prematurely if start_positions is null
        if start_positions[0] is None and start_positions[1] is None:
            # print "No start positions discovered"
            self.lanes = [None, None]
        else:
            # Create and slide windows
            self.assemble_lane_windows(start_positions)

            if case == 3:         
                # Calculate lane width from newly discovered lanes
                self.add_lane_width(self.lanes)

            # Extract lane info from the lanes
            results = self._extract_lane_info()

        # Draw the lanes
        cv2.imshow("drawn", self.birdseye_img)
        cv2.waitKey(1)

        return results