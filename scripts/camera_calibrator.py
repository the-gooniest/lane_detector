import argparse
from chessboard import ChessBoard
import cv2
import os
import pickle
import re


class CameraCalibrator:

    def __init__(self):
        print "Initialized CameraCalibrator"
        self.images_dir = '../calibration_images/front/'

    def calibrate(self):
        print "Generating camera calibration data..."
        chessboards = []

        image_names = [
            name for name in os.listdir(self.images_dir)
            if os.path.isfile(os.path.join(self.images_dir, name))
        ]
        image_names.sort(
            key=lambda x: int(re.search('\d+', x).group()), reverse=False)

        for name in image_names:
            print 'Calibrating ' + name
            this_path = self.images_dir + name
            chessboard = ChessBoard(path=this_path, nx=7, ny=6, generation=True)
            chessboards.append(chessboard)

        points, corners, shape = [], [], chessboards[0].dimensions

        for chessboard in chessboards:
            if chessboard.has_corners:
                points.append(chessboard.object_points)
                corners.append(chessboard.corners)

        r, matrix, distortion_coef, rv, tv = cv2.calibrateCamera(
            points, corners, shape, None, None)

        self.calibration_data = {
            "camera_matrix": matrix,
            "distortion_coefficient": distortion_coef
        }
        """
        for chessboard in chessboards:
            if chessboard.has_corners:
                 save_image(chessboard.image_with_corners(), "corners", chessboard.i)

            if chessboard.can_undistort:
                 save_image(chessboard.undistorted_image(), "undistortedboard", chessboard.i)
        """

    def save_calibration_data(self):
        print "Saving calibration data..."
        pickle.dump(self.calibration_data,
                    open("camera_calibration_data.p", "wb"))


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process images for lane data')
    #TODO: add args
    return parser.parse_args()


def main():
    args = parse_arguments()
    calibrator = CameraCalibrator()
    calibrator.calibrate()
    calibrator.save_calibration_data()


if __name__ == "__main__":
    main()
