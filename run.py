
from imutils.video import VideoStream
from board import WhiteboardDetector

import imutils
import cv2
import time

import argparse

detector = WhiteboardDetector(False, True)

def run_video(source = 0):
    vs = VideoStream(src=source).start()
    time.sleep(2.0)

    isLocked = False

    while True:
        frame = vs.read()
        
        board, corners = detector.crop_board(frame, isLocked)
        cv2.imshow('whiteboard', board)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):
            isLocked = ~isLocked

    cv2.destroyAllWindows()
    vs.stop()

def run_image_test():
    frame = cv2.imread('test-data/B87D0402-BE24-4115-B92C-1768EC139DFE.JPG')
    # frame = cv2.imread('test-data/76A8B258-7DE1-435D-8686-4E788990A6E9.JPG') # Bad Work
    # frame = cv2.imread('test-data/31B8DBD6-A853-401A-B8B5-D17B44DFAF73.JPG') 
    # frame = cv2.imread('test-data/IMG_20200526_121829.jpg') # Work, but transform Issue
    # frame = cv2.imread('test-data/bd.jpg') # No Work

    # frame = cv2.imread('test-data/7517BF67-2D28-46A4-A44A-1BA4B7F01135.JPG') # Work, but transform Issue
    # frame = cv2.imread('test-data/3558B111-2149-40B5-BB55-ACDB939A3B25.JPG') # Bad work
    # frame = cv2.imread('test-data/IMG_20200528_103005.jpeg') # Bad Work, perfect with 7 blur
    # frame = cv2.imread('test-data/IMG_20200528_104019.jpeg') # Bad Work, perfect with 7 or 9 blur
    # frame = cv2.imread('test-data/IMG_20200527_152304.jpeg') # No Work
    # frame = cv2.imread('test-data/4AC836AA-9D15-48DD-96FB-532DAD891676.JPG') # Work

    # frame = cv2.imread('test-data/1849049E-B847-4AF0-848F-EFC02B9E525F.JPG') # No Work
    # frame = cv2.imread('test-data/IMG_0579.jpeg') # Bad Work
    # frame = cv2.imread('test-data/IMG_20200527_112335.jpeg') # Work (but not perfect)

    # frame = cv2.imread('test-data/1.jpeg') # Work
    # frame = cv2.imread('6.jpeg') # Bad Work
    # frame = cv2.imread('board.jpg') # Bad Work

    # frame = cv2.imread('test-data/9E12899A-9CEE-4E7D-9BB3-A865D11D2EA5.JPG') # Work
    # frame = cv2.imread('test-data/A6A01BCF-F4A2-4983-99EA-28831ED806AD.JPG') # No Work
    # frame = cv2.imread('test-data/A1C83073-A742-4304-A3D3-0481BAC672A0.JPG') # No Woks

    board, corners = detector.crop_board(frame)
    cv2.imshow('whiteboard', board)

    print(corners)

    cv2.waitKey()
 
def run_image(path):
    frame = cv2.imread(path)
    board, corners = detector.crop_board(frame)
    cv2.imwrite('output.jpeg', board)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script that captures & crops the whiteboard')
    parser.add_argument('--path', type=str, help="Path to image if Image")

    args = parser.parse_args()

    print(args.path)

    if args.path is None:
        run_video()
    else:
        run_image(args.path)
