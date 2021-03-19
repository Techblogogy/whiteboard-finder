from imutils.video import VideoStream
import imutils
import cv2
import sys
import time
import cv2.aruco as aruco
import numpy as np
import socketio
import base64


def extract_corners(frame):
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

    board_corners = np.zeros((4, 2), dtype = "float32")

    np_corners = np.array(corners)
    np_corners = np_corners.reshape(4*4, 2)

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

def transform_image(frame, corners):
    if corners is None:
        return frame
    
    # TODO: Figure out if this [ rect *= ratio ] is needed
    # this will probably come in handy if board / image ratio is very different

    (tl, tr, br, bl) = corners

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
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

def send_data(frame):
    img = cv2.imencode('.jpg', frame)[1].tobytes()
    img = base64.b64encode(img).decode('utf-8')
    img_data = "data:image/jpeg;base64,{}".format(img)

    sio.emit('cv2server', {
        'image': img_data,
        'text': ''
    })

# sio = socketio.Client()

# @sio.on('change_mode', namespace='/cv')
# def on_message(data):
#     global bw_mode
#     bw_mode = not bw_mode

#     print(bw_mode)

# server_url = 'http://pathfinder-board.eastus.azurecontainer.io:5001/'
# sio.connect(server_url, transports=['websocket'], namespaces=['/cv'])

vs = VideoStream(src=0).start()
time.sleep(2.0)

old_points = None
last_update_t = time.time()
wait_t = 0 #2
bw_mode = True

# cv2.namedWindow("output", cv2.WND_PROP_FULLSCREEN) 
# cv2.setWindowProperty("output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True:
    # frame = vs.read()
    frame = cv2.imread('IMG_0580.jpeg')

    new_height = 1000

    ratio = frame.shape[0] / new_height
    small_frame = imutils.resize(frame, height = new_height)




    points = extract_corners(frame)

    if points is None:
        points = old_points
    else:
        old_sum = old_points.sum() if old_points is not None else 0
        new_sum = points.sum()

        if new_sum in range(int(old_sum) - 5, int(old_sum) + 5):
            points = old_points
        else:
            old_points = points

    # draw_corner_points(frame, points)
    wrap = transform_image(frame, points)

    kernel_sharpening = np.array([
        [-1,-1,-1], 
        [-1, 9,-1],
        [-1,-1,-1]
    ])

    sharp = cv2.filter2D(wrap, -1, kernel_sharpening)
    gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
    (thresh, mask) = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)

    # final = cv2.bitwise_and(wrap, wrap, mask = mask)

    # Display the resulting frame
    # cv2.imshow('frame', frame)
    # cv2.imshow('gray', gray)
    # cv2.imshow('sharp', sharp)
    cv2.imshow('wrap', wrap)
    cv2.imshow('small', small_frame)
    # cv2.imshow('mask', mask)

    cv2.imwrite('bd.jpg', wrap)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()