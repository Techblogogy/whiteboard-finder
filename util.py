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
    
    @staticmethod
    def are_points_close(a, b, r = 25):
        if (a is None) or (b is None):
            return False
        
        sum_diff = abs(int(a.sum() - b.sum()))

        if (sum_diff <= r):
            return True
        else:
            return False
