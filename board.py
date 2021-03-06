
import imutils
import cv2
import sys
import cv2.aruco as aruco
import numpy as np
from util import Util

###### BIG BRAIN ALGORITHM IDEA ######
#
# 0. <- Play around with different values / test cases / contiditions
#
# 1. Try to detect using Aruco markers
#   1.1. (???) Instead of Aurco, detect bright markers in corner of boards
#   1.2. (???) Retro reflective material for IR camera
# 2. Try to detect using my board extraction algo
# 3. Try my board extraction algo, but with different parameters (??? calibration mode)
# 4. Try alternative extraction algo as last resort

class WhiteboardDetector:

    # TODO: Add debug mode on / off
    def __init__(self, debug = False, useAruco = True):
        self._isDebug = debug

        # Used for auto-detection, image threshold
        self._blockSize = 259 # 105
        self._C = -10 # -6

        self._blurSize = 5 # alt. 3, 7, 9 (depending on accuracy)

        self._sigmaValue = 0.02

        self._useAruco = useAruco

        self._oldCorners = None
    
    def crop_board(self, frame, isLocked = False):

        # Feature to lock detection in place (in case people walk in)
        if isLocked:
            print("Locked")
            wrap = self.transform_image(frame, self._oldCorners)
            return wrap, self._oldCorners

        # Detect by using markers or automatically
        # TODO: Think of a better way / logic to merge theese
        if (self._useAruco):
            corners = self.extract_corners_aruco(frame)
        else:
            corners = self.extract_corners_auto(frame)

        # Prevents wiggle during video
        if corners is None:
            corners = self._oldCorners
        elif Util.are_points_close(corners, self._oldCorners):
            corners = self._oldCorners
        else:
            self._oldCorners = corners

        # TODO: If less than 4 markers found, attemt to combine aruco + big brain

        wrap = self.transform_image(frame, corners)

        return wrap, corners
    
    # Extract Whiteboard Corners Using AR Markers 
    def extract_corners_aruco(self, frame):
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        parameters = aruco.DetectorParameters_create()
        parameters.minDistanceToBorder = 0
        # parameters.adaptiveThreshWinSizeMax = 500

        corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if (self._isDebug):
            frame = aruco.drawDetectedMarkers(frame, corners, ids)

        if len(corners) != 4:
            # print("Trackers not found")
            return None
        
        return self.get_corners_coords(corners, 16)

    # Automaticaly Extract Whiteboard Corners
    def extract_corners_auto(self, big_frame):
        frame, ratio = Util.resize_image(big_frame)

        ###### GRAYSCALE & BLUR IMAGE ######
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, self._blurSize) 

        # TODO: Alternative Bluring Strategies Test
        # gray = cv2.bilateralFilter(gray, 5, 17, 17) 
        # gray = cv2.GaussianBlur(gray, (7, 7), 3)

        ###### Extract B&W IMAGE ######
        bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self._blockSize, self._C)

        # TODO: Alternative Threshold Strategy
        # (thresh, bw) = cv2.threshold(gray, 158, 255, cv2.THRESH_BINARY)

        if (self._isDebug):
            cv2.imshow('gray', gray)
            cv2.imshow('bw', bw)

        ###### (Alt.) CANNY EDGE DETECTION ######

        # sigma = 0.33

        # v = np.median(bw)
        # lower = int(max(0, (1.0 - sigma) * v))
        # upper = int(min(255, (1.0 + sigma) * v))

        # edged = cv2.Canny(bw, lower, upper)
        # cv2.imshow('edged', edged)


        ###### (Alt.) HOUGH LINES DETECTION ######

        # lines = cv2.HoughLinesP(edged, 1, np.pi/180, 150, maxLineGap=500)
        # for line in lines:
        #     x1, y1, x2, y2 = line[0]
        #     cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)


        ###### CONTOUR DETECTION ######
        cnts = cv2.findContours(bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

        screenCnt = None
        maxArea = 0

        # loop over our contours
        for c in cnts:
            # TODO: Turn color into constant
            cv2.drawContours(frame, [c], -1, (0, 255, 0))

            peri = cv2.arcLength(c, True)

            # TODO: Play around with sigma value more, for edge cases
            approx = cv2.approxPolyDP(c, self._sigmaValue * peri, True)
            
            # TODO: Turn color into constant
            cv2.drawContours(frame, [approx], -1, (255, 0, 0))

            area = cv2.contourArea(approx)
            if area > maxArea:
                maxArea = area
            else:
                continue

            # TODO: ++ Experiment how to extract more than 4 points polygon (approximation)
            #   Idea: Do a thing simmilar to aruco (get_corners_coords) function if shape is more than 4
            #           + make sure that shape is big enough to be a whiteboard (e.g. not super small)
            # TODO: ++ Find the biggest countour, not just first one
            if len(approx) == 4:
                # TODO: Turn color into constant
                cv2.drawContours(frame, [approx], -1, (0, 0, 255))
                screenCnt = approx

                # break
            elif len(approx) > 4:
                screenCnt = self.get_corners_coords(approx)
                # print(screenCnt)
                # cv2.drawContours(frame, [screenCnt], -1, (0, 0, 255))

        corners = self.get_corners_coords(screenCnt)
        corners *= ratio

        if (self._isDebug):
            Util.draw_corner_points(big_frame, corners)
            cv2.imshow('frame', big_frame)
            cv2.imshow('small', frame)

        return corners

    # Extract 4 edge points from shape
    def get_corners_coords(self, corners, size = -1):
        board_corners = np.zeros((4, 2), dtype = "float32")

        np_corners = np.array(corners)
        # np_corners = np_corners.reshape(4*4, 2) # TODO: Auto adjust
        # np_corners = np_corners.reshape(4, 2)
        if (size == -1):
            np_corners = np_corners.reshape(len(corners), 2)
        else:
            np_corners = np_corners.reshape(size, 2)

        c_sum = np_corners.sum(axis = 1)
        board_corners[0] = np_corners[np.argmin(c_sum)] # Top Left
        board_corners[2] = np_corners[np.argmax(c_sum)] # Bottom Right

        c_dif = np.diff(np_corners, axis = 1)
        board_corners[1] = np_corners[np.argmin(c_dif)] # Top Right
        board_corners[3] = np_corners[np.argmax(c_dif)] # Bottom Left

        return board_corners

    # Straigten Whiteboard Image
    def transform_image(self, frame, corners):
        if corners is None:
            return frame
        
        (tl, tr, br, bl) = corners

        widthA = Util.calculate_distance(br, bl) #np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = Util.calculate_distance(tr, tl) #np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        
        heightA = Util.calculate_distance(tr, br) #np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = Util.calculate_distance(tl, bl) #np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        
        # TODO: Fix issue related to aspect ratio going wierd

        # === Max Method ===
        # maxWidth = max(int(widthA), int(widthB))
        # maxHeight = max(int(heightA), int(heightB))

        # === Min Method ===
        maxWidth = min(int(widthA), int(widthB))
        maxHeight = min(int(heightA), int(heightB))

        # TODO: Think about better logs solution
        print("Size: ", maxWidth, "x", maxHeight)

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        
        M = cv2.getPerspectiveTransform(corners, dst)
        warp = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

        return warp

    # TODO: Add image cleanup code
    def cleanup_image(self, frame):
        # kernel_sharpening = np.array([
        #     [-1,-1,-1], 
        #     [-1, 9,-1],
        #     [-1,-1,-1]
        # ])

        # sharp = cv2.filter2D(wrap, -1, kernel_sharpening)
        # gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
        # (thresh, mask) = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

        # # final = cv2.bitwise_and(wrap, wrap, mask = mask)

        # # Display the resulting frame
        # # cv2.imshow('frame', frame)
        # # cv2.imshow('gray', gray)
        # # cv2.imshow('sharp', sharp)
        # cv2.imshow('wrap', wrap)
        # cv2.imshow('small', small_frame)
        # # cv2.imshow('mask', mask)
        pass
