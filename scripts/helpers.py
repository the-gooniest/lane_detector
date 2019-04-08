import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.patches import Circle


def scale_abs(x, m=255):
    max_x = np.max(x)
    if max_x == 0:
        return x
    x = np.absolute(x)
    x = np.uint8(m * x / np.max(x))
    return x


def roi(gray, mn=125, mx=1200):
    m = np.copy(gray) + 1
    m[:, :mn] = 0
    m[:, mx:] = 0
    return m


def save_image(img, name, i):
    path = "output_images/" + name + str(i) + ".jpg"
    cv2.imsave(path, img)


def show_images(imgs, per_row=3, per_col=2, W=10, H=5, tdpi=80):
    fig, ax = plt.subplots(per_col, per_row, figsize=(W, H), dpi=tdpi)
    ax = ax.ravel()

    for i in range(len(imgs)):
        img = imgs[i]
        ax[i].imshow(img)

    for i in range(per_row * per_col):
        ax[i].axis('off')


def show_dotted_image(this_image,
                      points,
                      thickness=5,
                      color=[255, 0, 255],
                      d=15):
    image = this_image.copy()

    cv2.line(image, points[0], points[1], color, thickness)
    cv2.line(image, points[2], points[3], color, thickness)

    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(image)

    for (x, y) in points:
        dot = Circle((x, y), d)
        ax.add_patch(dot)

    plt.show()


def show_single_image(image):
    if image is None:
        raise RuntimeError("Attempted to show empty image matrix")
    cv2.imshow('image', image)
    cv2.waitKey(0)


def crop_image(image, percent):
    """Crop a percentage of each side of the given image"""
    x = int(self.image_size[0] * percent)
    y = int(self.image_size[1] * percent)
    w = int(self.image_size[0] * (1 - percent))
    h = int(self.image_size[1] * (1 - percent))
    return image[y:h, x:w]


def resize_image(image, scale_factor):
    height, width = image.shape[:2]
    scaled_image = cv2.resize(
        image, (int(width * scale_factor), int(height * scale_factor)))
    return scaled_image


def local_coordinates(shape, pt):
    x, y = pt
    width, height = shape

    x_scale = 0.008403361344538
    # used for only 712 pixel in y
    y_scale = 0.031250000000000
    # used for only 712 pixel in y

    local_x = (x - width / 2) * x_scale
    local_y = (height - y) * y_scale
    local_out = (local_x, local_y)

    return local_out
