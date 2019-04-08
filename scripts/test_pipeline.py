import argparse
import cv2
from helpers import show_single_image
import numpy as np
import os
from pipeline import DetectionPipeline
import pickle
import pprint
import time


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process images for lane data')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')
    parser.add_argument('-s', '--single-image', required=False, dest='single')
    parser.add_argument(
        '-f',
        '--image-dir',
        required=False,
        dest='dir',
        help='Run pipeline with images from the provided directory path')

    parser.set_defaults(verbose=False)
    return parser.parse_args()


def resize_image(img, target_size):
    target_size = float(target_size)
    height = img.shape[0]
    width = img.shape[1]
    if height > width:
        scale_factor = target_size / height
        img = cv2.resize(
            img, (int(width * scale_factor), int(height * scale_factor)))
    else:
        scale_factor = target_size / width
        img = cv2.resize(
            img, (int(width * scale_factor), int(height * scale_factor)))
    return img


def main():
    args = parse_arguments()
    calibration_data = pickle.load(open("camera_calibration_data.p", "rb"))

    if args.single:
        image_paths = [args.single]
    elif args.dir:
        image_paths = os.listdir(args.dir)
        image_paths = [os.path.join(args.dir, f) for f in sorted(image_paths)]
        if len(image_paths) == 0:
            print "No files in the provided directory"

    img = cv2.imread(image_paths[0])
    if img is None:
        print "Invalid image file", image_paths[0]

    pipeline = DetectionPipeline((512, 512), calibration_data)

    for image_path in image_paths:
        img = cv2.imread(image_path)

        if img is None:
            print "Invalid image file", image_path
            continue

        start_time = time.time()
        img = resize_image(img, 512)

        result = pipeline.run(img)
        #if result is not None:
        #print result['lane_heading']
        #pprint.pprint(result)


if __name__ == "__main__":
    main()
