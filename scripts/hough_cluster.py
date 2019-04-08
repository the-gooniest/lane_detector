import cv2
from helpers import show_single_image
import numpy as np
import matplotlib.pyplot as plt


class HoughCluster:

    def __init__(self, shape):
        # The highest y point in the cluster
        self.r_point = None
        self.x_points = []
        self.y_points = []

        # Shape of the image the cluster points were extracted from
        self.img_shape = shape
        self.img_width = shape[1]
        self.img_height = shape[0]

        # Linear regression class members
        self.calculated_fit = False
        self.direction = None
        self.intercept = None
        self.fit_function = None

        # Whether this cluster was formed from two clusters merging
        self.merged = False

        # minimum distance between interpolated hough points
        self.min_dist = 20
        self.min_dist_sqr = self.min_dist * self.min_dist

    def add_point(self, point):
        """Add a hough line point to this cluster"""

        self.r_point = point
        self.x_points.append(point[0])
        self.y_points.append(point[1])

    def close_enough(self, point):
        """Checks if the given point is close enough to this cluster's
           reprentative point to be added to the cluster"""

        if self.mag2(point, self.r_point) < self.min_dist_sqr:
            self.add_point(point)
            return True
        return False

    def mag2(self, point1, point2):
        """The squared distance between two points"""

        return np.square(point1[0] - point2[0]) + np.square(
            point1[1] - point2[1])

    def center(self):
        """The mean center point of this cluster"""

        mean_x = int(np.array(self.x_points).mean())
        mean_y = int(np.array(self.y_points).mean())
        return (mean_x, mean_y)

    def fit_point(self, y):
        """Returns an x value for some y value along the cluster's fit line"""

        if self.fit_function is not None:
            return int(self.fit_function(y))
        else:
            return None

    def lane_start(self):
        """Returns the pixel x value of the fit line of this cluster"""

        if self.calculated_fit:
            return int(self.fit_function(
                self.img_height)) - (self.img_width / 2)
        else:
            return None

    def lane_angle(self):
        """Returns the slope of this cluster's fit line"""

        if self.calculated_fit:
            return self.direction
        else:
            return None

    def fit(self):
        """Fit a regression line to the points in this cluster"""

        if not self.calculated_fit:
            fit = np.polyfit(self.y_points, self.x_points, 1)
            self.fit_function = np.poly1d(fit)
            self.direction = np.arctan(fit[0])
            self.intercept = int(self.fit_function(0))
            self.calculated_fit = True

    def combine_clusters(self, cluster):
        """Merge two clusters if their regression lines are similar"""

        self.fit()
        cluster.fit()
        if np.absolute(self.direction - cluster.direction) < np.pi / 16 and \
            np.absolute(self.intercept - cluster.intercept) < 50:
            new_cluster = HoughCluster(self.img_shape)
            new_cluster.x_points = self.x_points + cluster.x_points
            new_cluster.y_points = self.y_points + cluster.y_points
            new_cluster.merged = True
            return new_cluster
        return None
