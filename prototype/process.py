from imutils.video import VideoStream
import imutils
import cv2
import sys
import time
import cv2.aruco as aruco
import numpy as np
import socketio
import base64
import random


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


def extract_corners_aruco(frame):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
    parameters = aruco.DetectorParameters_create()
    parameters.minDistanceToBorder = 0
    # parameters.adaptiveThreshWinSizeMax = 500

    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # Uncomend to draw border around markers
    frame = aruco.drawDetectedMarkers(frame, corners, ids)

    if len(corners) != 4:
        # print("Trackers not found")
        return

    return get_corners_coords(corners)

def extract_corners_auto(frame):

    ###### GRAYSCALE & BLUR IMAGE ######

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bilateralFilter(gray, 5, 17, 17)

    # gray = cv2.medianBlur(gray, 3) 
    gray = cv2.medianBlur(gray, 5) 
    # gray = cv2.medianBlur(gray, 7) # Less Accurate or more accurate, it depends
    # gray = cv2.medianBlur(gray, 9) # Less Accurate or more accurate, it depends

    # gray = cv2.GaussianBlur(gray, (7, 7), 3)

    ###### Extract B&W IMAGE ######

    # (thresh, bw) = cv2.threshold(gray, 158, 255, cv2.THRESH_BINARY)

    # TODO: Play around with blockSize & C parameters
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 259, -10)
    # bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 105, -6)

    # Show Image For Debug
    cv2.imshow('gray', gray)
    cv2.imshow('bw', bw)

    ###### CANNY EDGE DETECTION ######

    # sigma = 0.33

    # v = np.median(bw)
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))

    # edged = cv2.Canny(bw, lower, upper)
    # cv2.imshow('edged', edged)


    ###### HOUGH LINES DETECTION (instead of contour) ######

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
        cv2.drawContours(frame, [c], -1, (0, 255, 0))

        peri = cv2.arcLength(c, True)
        # TODO: Play around with sigma value more, for edge cases
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        cv2.drawContours(frame, [approx], -1, (255, 0, 0))

        area = cv2.contourArea(approx)
        if area > maxArea:
            maxArea = area
        else:
            continue

        # TODO: Experiment how to extract more than 4 points polygon (approximation)
        #   Idea: Do a thing simmilar to aruco (get_corners_coords) function if shape is more than 4
        #           + make sure that shape is big enough to be a whiteboard (e.g. not super small)
        # TODO: Find the biggest countour, not just first one
        if len(approx) == 4:
            cv2.drawContours(frame, [approx], -1, (0, 0, 255))
            screenCnt = approx

            # break
        elif len(approx) > 4:
            screenCnt = get_corners_coords(approx)
            # print(screenCnt)
            # cv2.drawContours(frame, [screenCnt], -1, (0, 0, 255))

    return screenCnt

def extract_corners_auto2(frame):

    ##### INIT VARS #####
    precision = 0.12
    canny1 = 30
    canny2 = 265

    biggestAreaContours = None
    largestAreaFound = 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 17, 17)
    edged = cv2.Canny(gray, canny1, canny2)

    cv2.imshow('gray', gray)
    cv2.imshow('edged', edged)

    # FIND THE SCREEN
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    area = None

    # Use cached contours, if any
    screenCnt = biggestAreaContours

    # loop over our contours WHICH ARE NOW SORTED IN ORDER FROM BIGGEST DOWN
    for c in cnts:
        cv2.drawContours(frame, [c], -1, (0, 255, 0))

        # approximate the contour
        peri = cv2.arcLength(c, True)
        area = cv2.contourArea(c)

        # There is no way a tiny board can happen so limit min size
        if (area < largestAreaFound-100):
            continue
        
        # This area is smaller than previously cached one, don't use it
        elif (area < largestAreaFound):
            continue

        approxOutline = cv2.approxPolyDP(c, precision * peri, True)

        cv2.drawContours(frame, [approxOutline], -1, (255, 0, 0))

        # if our approximated contour has four or more points, then
        # we can assume that we have found our screen and can do further
        # processing if needed
        if len(approxOutline) >= 4:
            
            # qcif we made it this far it is a good contour and so we can save it
            largestAreaFound = area
            biggestAreaContours = approxOutline

            cv2.drawContours(frame, [approxOutline], -1, (0, 0, 255))

            screenCnt = approxOutline
            break

    if screenCnt is None:
        print('Not Found')

    return screenCnt

def get_corners_coords(corners):
    board_corners = np.zeros((4, 2), dtype = "float32")

    np_corners = np.array(corners)
    # np_corners = np_corners.reshape(4*4, 2) <- # TODO: Auto adjust
    # np_corners = np_corners.reshape(4, 2)
    np_corners = np_corners.reshape(len(corners), 2)

    c_sum = np_corners.sum(axis = 1)
    board_corners[0] = np_corners[np.argmin(c_sum)] # Top Left
    board_corners[2] = np_corners[np.argmax(c_sum)] # Bottom Right

    c_dif = np.diff(np_corners, axis = 1)
    board_corners[1] = np_corners[np.argmin(c_dif)] # Top Right
    board_corners[3] = np_corners[np.argmax(c_dif)] # Bottom Left

    return board_corners

def draw_corner_points(frame, corners):
    if corners is None:
        return
    
    for corner in corners:
        cv2.circle(frame, tuple(corner), 4, (0, 0, 255), -1)

def calculate_distance(a, b):
    return np.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))

def transform_image(frame, corners):
    if corners is None:
        return frame
    
    # TODO: Figure out if this [ rect *= ratio ] is needed
    # this will probably come in handy if board / image ratio is very different

    (tl, tr, br, bl) = corners

    widthA = calculate_distance(br, bl) #np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = calculate_distance(tr, tl) #np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    
    heightA = calculate_distance(tr, br) #np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = calculate_distance(tl, bl) #np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
    # TODO: Fix wierd but related to aspect ratio going weird

    # maxWidth = max(int(widthA), int(widthB))
    # maxHeight = max(int(heightA), int(heightB))
    maxWidth = min(int(widthA), int(widthB))
    maxHeight = min(int(heightA), int(heightB))

    print("Size: ", maxWidth, "x", maxHeight)

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(corners, dst)
    warp = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))

    return warp

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def nothing():
    pass

frame = cv2.imread('B87D0402-BE24-4115-B92C-1768EC139DFE.JPG')
# frame = cv2.imread('76A8B258-7DE1-435D-8686-4E788990A6E9.JPG') # Bad Work
# frame = cv2.imread('31B8DBD6-A853-401A-B8B5-D17B44DFAF73.JPG') 
# frame = cv2.imread('IMG_20200526_121829.jpg') # Work, but transform Issue
# frame = cv2.imread('bd.jpg') # No Work

# frame = cv2.imread('7517BF67-2D28-46A4-A44A-1BA4B7F01135.JPG') # Work, but transform Issue
# frame = cv2.imread('3558B111-2149-40B5-BB55-ACDB939A3B25.JPG') # Bad work
# frame = cv2.imread('IMG_20200528_103005.jpeg') # Bad Work, perfect with 7 blur
# frame = cv2.imread('IMG_20200528_104019.jpeg') # Bad Work, perfect with 7 or 9 blur
# frame = cv2.imread('IMG_20200527_152304.jpeg') # No Work
# frame = cv2.imread('4AC836AA-9D15-48DD-96FB-532DAD891676.JPG') # Work

# frame = cv2.imread('1849049E-B847-4AF0-848F-EFC02B9E525F.JPG') # No Work
# frame = cv2.imread('IMG_0579.jpeg') # Bad Work
# frame = cv2.imread('IMG_20200527_112335.jpeg') # Work (but not perfect)

# frame = cv2.imread('1.jpeg') # Work
# frame = cv2.imread('6.jpeg') # Bad Work
# frame = cv2.imread('board.jpg') # Bad Work

# frame = cv2.imread('9E12899A-9CEE-4E7D-9BB3-A865D11D2EA5.JPG')
# frame = cv2.imread('A6A01BCF-F4A2-4983-99EA-28831ED806AD.JPG')
# frame = cv2.imread('A1C83073-A742-4304-A3D3-0481BAC672A0.JPG')



# blockSize = 3
# C = 0
# blur = 3
# while True:

#     print('Block Size: ', blockSize, ' C: ', C, ' Blur: ', blur)

#     gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.medianBlur(gray, blur)
#     bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize, C)
#     # bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize, C)

#     cv2.imshow('gray', gray)
#     cv2.imshow('bw', bw)

#     key = cv2.waitKey(1) & 0xFF

#     if key == ord("q"):
#         break
#     elif key == ord("w"):
#         blockSize += 2
#     elif key == ord("s"):
#         blockSize -= 2
#     elif key == ord("a"):
#         C += 1
#     elif key == ord("d"):
#         C -= 1
#     elif key == ord("r"):
#         blur += 2
#     elif key == ord("f"):
#         blur -= 2



# cv2.waitKey()

vs = VideoStream(src=2).start()
time.sleep(2.0)

while True:
    frame = vs.read()

    new_height = 300
    ratio = frame.shape[0] / new_height

    small_frame = imutils.resize(frame, height = new_height)

    screenCnt = extract_corners_auto(small_frame)
    corners = get_corners_coords(screenCnt)
    corners *= ratio
    draw_corner_points(frame, corners)
    wrap = transform_image(frame, corners)

    cv2.imshow('frame', frame)
    cv2.imshow('wrap', wrap)

    cv2.imshow('small', small_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()

# old_points = None


# while True:
#     # frame = vs.read()
#     frame = cv2.imread('IMG_0580.jpeg')

#     new_height = 1000

#     ratio = frame.shape[0] / new_height
#     small_frame = imutils.resize(frame, height = new_height)




#     points = extract_corners_aruco(frame)

#     if points is None:
#         points = old_points
#     else:
#         old_sum = old_points.sum() if old_points is not None else 0
#         new_sum = points.sum()

#         if new_sum in range(int(old_sum) - 5, int(old_sum) + 5):
#             points = old_points
#         else:
#             old_points = points

#     # draw_corner_points(frame, points)
#     wrap = transform_image(frame, points)

#     kernel_sharpening = np.array([
#         [-1,-1,-1], 
#         [-1, 9,-1],
#         [-1,-1,-1]
#     ])

#     sharp = cv2.filter2D(wrap, -1, kernel_sharpening)
#     gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
#     (thresh, mask) = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

#     # final = cv2.bitwise_and(wrap, wrap, mask = mask)

#     # Display the resulting frame
#     # cv2.imshow('frame', frame)
#     # cv2.imshow('gray', gray)
#     # cv2.imshow('sharp', sharp)
#     cv2.imshow('wrap', wrap)
#     cv2.imshow('small', small_frame)
#     # cv2.imshow('mask', mask)

#     cv2.imwrite('bd.jpg', wrap)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord("q"):
#         break

# cv2.destroyAllWindows()
# vs.stop()