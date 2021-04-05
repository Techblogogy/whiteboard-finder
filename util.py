import imutils
import numpy as np
import cv2

class Util:

    # Resize Image By Height + Keep Aspect Ratio
    @staticmethod
    def resize_image(frame, new_height = 300):
        ratio = frame.shape[0] / new_height
        small_frame = imutils.resize(frame, height = new_height)

        return small_frame, ratio

    # Draw corners for debug purposes
    @staticmethod
    def draw_corner_points(frame, corners):
        if corners is None:
            return
        
        for corner in corners:
            cv2.circle(frame, tuple(corner), 4, (0, 0, 255), -1)

    # Calculate distance between 2 points
    @staticmethod
    def calculate_distance(a, b):
        return np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))


# TODO: Code to compare points proximity
# if points is None:
#     points = old_points
# else:
#     old_sum = old_points.sum() if old_points is not None else 0
#     new_sum = points.sum()

#     if new_sum in range(int(old_sum) - 5, int(old_sum) + 5):
#         points = old_points
#     else:
#         old_points = points


