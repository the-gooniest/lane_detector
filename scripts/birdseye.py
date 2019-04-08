import cv2
import numpy as np


class BirdsEye:

    def __init__(self, source_points, dest_points):
        self.src_points = np.array(source_points, np.float32)
        self.dest_points = np.array(dest_points, np.float32)
        self.warp_matrix = cv2.getPerspectiveTransform(self.src_points,
                                                       self.dest_points)

    def birdseye_view(self, ground_image):
        """Transforms ground_image to a birdseye view"""

        shape = (self.dest_points[2][0] - self.dest_points[1][0],
                 self.dest_points[2][1] - self.dest_points[3][1])

        warp_image = cv2.warpPerspective(
            ground_image, self.warp_matrix, shape, flags=cv2.INTER_LINEAR)

        return warp_image
