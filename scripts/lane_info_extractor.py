import cv2
from helpers import show_single_image
from hough_cluster import HoughCluster
import numpy as np
import matplotlib.pyplot as plt


class LaneInfoExtractor:

    def __init__(self):
        self.lane_widths = []

    def cluster_hough_points(self, points):
        """Categorize points from hough lines into lane clusters"""

        # Sort points on y axis
        points = sorted(points, key=lambda k: [k[1], k[0]])

        # Prepare for clustering process
        initial_cluster = HoughCluster(self.img.shape)
        initial_cluster.add_point(points[0])
        clusters = [initial_cluster]

        # Sort points into clusters
        for point in points[1:]:
            found_cluster = False
            for cluster in clusters:
                if cluster.close_enough(point):
                    found_cluster = True
                    break
            if not found_cluster:
                new_cluster = HoughCluster(self.img.shape)
                new_cluster.add_point(point)
                clusters.append(new_cluster)

        # Remove small clusters
        good_clusters = []
        for cluster in clusters:
            if len(cluster.x_points) > 15:
                good_clusters.append(cluster)
        clusters = good_clusters

        # Merge similar clusters
        merged_clusters = False
        while True:
            merged_clusters = False
            for i in range(0, len(clusters) - 1):
                for j in range(i + 1, len(clusters)):
                    new_cluster = clusters[i].combine_clusters(clusters[j])
                    if new_cluster is not None:
                        del clusters[i]
                        del clusters[j - 1]
                        clusters.append(new_cluster)
                        merged_clusters = True
                        break
                if merged_clusters:
                    break
            if not merged_clusters:
                break

        # Draw cluster regression lines
        for cluster in clusters:
            cluster.fit()
            point1 = (int(cluster.fit_function(0)), 0)
            point2 = (int(cluster.fit_function(self.width)), self.width)
            self.rgb_copy = cv2.line(self.rgb_copy, point1, point2, (255, 0, 0),
                                     2)

        return clusters

    def extract_lane_info(self, masked_img, rgb_image):
        """Extract lane center, angle, and solid/dash info from a lane filtered image"""

        # Setup class members
        self.img = masked_img
        self.rgb_copy = np.copy(rgb_image)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]

        lines = cv2.HoughLinesP(self.img, 1, np.pi / 180, 25, maxLineGap=60)

        # Interpolate points from the detected hough lines
        points = []
        resolution = 8
        if lines is None:
            return None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            self.rgb_copy = cv2.line(self.rgb_copy, (x1, y1), (x2, y2),
                                     (0, 255, 0), 2)
            diff = x2 - x1
            if diff != 0:
                slope = np.absolute((y2 - y1) / diff)
                if slope < 0.4:
                    continue
            for i in range(0, resolution + 1):
                t = float(i) / resolution
                x = int(t * x1 + (1 - t) * x2)
                y = int(t * y1 + (1 - t) * y2)
                point = (x, y)
                points.append(point)
                #self.rgb_copy = cv2.circle(self.rgb_copy, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow("lines", self.rgb_copy)
        return None
        """
        # Cluster the interpolated hough line points
        clusters = self.cluster_hough_points(points)

        # TODO: handle edge case where there is zero or one cluster
        if len(clusters) <= 1:
            return None

        # Prepare result dictionary
        result = {}
        result["lane_center_offset"] = 0
        result["lane_angle"] = 0
        result['lane_width'] = 0
    
        # Find the cluster nearest to the center
        clusters = sorted(clusters, key=lambda k: k.lane_start())
        closest_cluster_index = 0
        for i, cluster in enumerate(clusters):
            lane_start_mag = np.absolute(cluster.lane_start())
            other_start_mag = np.absolute(clusters[closest_cluster_index].lane_start())
            if lane_start_mag < other_start_mag:
                closest_cluster_index = i

        # Calculate the average width of the lane
        center_lane_start = clusters[closest_cluster_index].lane_start()
        avg_lane_width = self.width / 6
        if len(self.lane_widths) > 0:
            avg_lane_width = int(np.average(self.lane_widths))
        max_lane_width = avg_lane_width * 1.2

        # Calculate the center of the lane using the clusters
        lane_width = None
        other_cluster_index = None
        lane_center_offset = 0
        if center_lane_start > 0:
            if closest_cluster_index > 0:
                lane_width = center_lane_start - clusters[closest_cluster_index - 1].lane_start()
                if lane_width < max_lane_width:
                    other_cluster_index = closest_cluster_index - 1
                    lane_center_offset = center_lane_start - (lane_width / 2)
                else:
                    lane_center_offset = center_lane_start - avg_lane_width / 2
            else:
                lane_center_offset = center_lane_start - avg_lane_width / 2
        else:
            if closest_cluster_index < len(clusters) - 1:
                lane_width = clusters[closest_cluster_index + 1].lane_start() - center_lane_start
                if lane_width < max_lane_width:
                    other_cluster_index = closest_cluster_index + 1
                    lane_center_offset = center_lane_start + (lane_width / 2)
                else:
                    lane_center_offset = center_lane_start + avg_lane_width / 2
            else:
                lane_center_offset = center_lane_start + avg_lane_width / 2

        # Record lane_width to help calculate the average lane width
        if lane_width is not None and lane_width < max_lane_width:
                if len(self.lane_widths) == 5:
                    self.lane_widths.pop()
                self.lane_widths.append(lane_width)
        result['lane_width'] = avg_lane_width

        # Record lane center offset
        PIXELS_TO_METERS_FACTOR = .92 / 16.5
        scaled_center_offset = lane_center_offset * PIXELS_TO_METERS_FACTOR
        result["lane_center_offset"] = lane_center_offset * PIXELS_TO_METERS_FACTOR

        # Record lane angle
        if other_cluster_index is not None:
            result["lane_angle"] = clusters[closest_cluster_index].lane_angle() + \
                clusters[other_cluster_index].lane_angle() / 2
        else:
            result["lane_angle"] = clusters[closest_cluster_index].lane_angle()
        
        # Draw lane center
        lane_pixel_center = lane_center_offset + self.width/2
        cv2.circle(self.rgb_copy, (lane_pixel_center, self.height - 4), 8, (0, 0, 255), -1)

        # Indicate the center of the image
        cv2.circle(self.rgb_copy, (self.width/2, self.height - 4), 4, (0, 255, 255), -1)

        # Show extracted info
        cv2.imshow('points', self.rgb_copy)
        cv2.waitKey(1)

        return result
        """
