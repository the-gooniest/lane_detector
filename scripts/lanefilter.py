import cv2
from helpers import show_single_image
import matplotlib.pyplot as plt
import numpy as np


class LaneFilter:

    def __init__(self):
        self.lane_widths = []

    def apply(self, rgb_image):
        """Initiates the filtering process"""

        self.rgb_image = rgb_image

        # separate hls (hue, lightness, and saturation) channels
        self.hls = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HLS)
        self.h = self.hls[:, :, 0]
        self.l = self.hls[:, :, 1]
        self.s = self.hls[:, :, 2]

        # Create white color mask
        self.white_color_mask = self.create_white_color_mask()
        #self.white_color_mask = self.remove_noise(self.white_color_mask)

        # Do canny edge detection
        #canny_img = self.create_canny_mask()

        # Combine the results of the two filters
        #filtered_img = cv2.bitwise_and(canny_img, self.white_color_mask)
        #filtered_img = self.dialate_image(filtered_img)
        return self.white_color_mask

    def dialate_image(self, img):
        """Fill gaps between filtered pixels"""

        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        open_img = cv2.dilate(img, kernel, iterations=1)
        return open_img

    def remove_noise(self, img):
        """Remove noise via morphology"""

        kernel1 = np.ones((1, 1), np.uint8)
        open_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
        return open_img

    def greyscale_threshold(self, img):
        # Create a histogram of the pixel lightness values
        bin_counts, bin_edges = np.histogram(img, range(0, 256))
        bin_counts[0] = 0

        """
        plt.plot(range(0,255), bin_counts)
        plt.pause(0.05)
        plt.clf()
        """

        # White pixels are only a small subset of many darker pixels in a lane image.
        # We look for the first upward slope of the histogram to find the threshold
        # point between the white lane pixels and the darker road pixels.
        """
        hist_max = np.max(bin_counts)
        for i in reversed(range(0, len(bin_counts))):
            slope = (bin_counts[i - 5] - bin_counts[i]) / float(hist_max)
            if slope > 0.06:
                return i + 20
        """

        # Find the end of the lane pixels by finding a histogram value
        # that's proportionally significant to the maximum histogram value.
        hist_max = np.max(bin_counts)
        for i in reversed(range(0, len(bin_counts) - 20)):
            normalized_hist_value = bin_counts[i] / float(hist_max)
            if normalized_hist_value > 0.05:
                return np.clip(i + 20, 100, 235)

    def create_white_color_mask(self):
        """Filters out everything but white pixels"""

        light_thresh = self.greyscale_threshold(self.l)

        # Use the threshold value just calculated and a separate saturation
        # threshold to filter out the lane pixels in the image.

        lane_pixels = np.zeros_like(self.l)
        lightness_condition = self.l > light_thresh
        lane_pixels[lightness_condition] = 1

        return lane_pixels

    def create_canny_mask(self):
        """Filters image for semi-vertical lines"""

        # Blur everything but the light pixels
        blurred_img = cv2.bilateralFilter(self.l, 9, 75, 75)
        masked_lightness = cv2.bitwise_and(
            self.l, self.l, mask=self.white_color_mask)
        mask_inv = 1 - self.white_color_mask
        blurred_img = cv2.bitwise_and(blurred_img, blurred_img, mask=mask_inv)
        blurred_img = cv2.add(blurred_img, masked_lightness)

        # Perform canny edge detection on the blurred image
        filtered_img = cv2.Canny(blurred_img, 36, 53)

        # Remove birds eye edge detection artifacts
        height, width = filtered_img.shape
        line_offset = width * 1 / 3
        cv2.line(filtered_img, (0, 0), (line_offset, height), 0, 20)
        cv2.line(filtered_img, (width, 0), (width - line_offset, height), 0, 20)
        ret, thresh = cv2.threshold(filtered_img, 1, 1, cv2.THRESH_BINARY)
        return thresh
