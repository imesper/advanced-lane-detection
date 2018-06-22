import math
import os
import time
import sobel
import color_masks
import matplotlib.image as mpimg
from line import Line
import numpy as np
from lane_detection import LaneDetection
import convolution
from moviepy.editor import VideoFileClip
from camera_calibration import camera_calibration, undistort_image, test_calibration
import cv2


def get_avarage_xfitted(fitsx, smooth=5):
    average = 0
    if len(fitsx) == smooth:
        for fitx in fitsx:
            average += fitx
        average /= 5
    return average


rightLine = [0] * 4
leftLine = [0] * 4
count = 0
smooth = 5

mtx, dist = camera_calibration()

leftLines = []
rightLines = []
lastLeftLine = Line()
lastRightLine = Line()

cap = cv2.VideoCapture('./project_video.mp4')
# cap = cv2.VideoCapture('./challenge_video.mp4')
# cap = cv2.VideoCapture('./harder_challenge_video.mp4')

detection = LaneDetection(nwindows=20)

fourcc = cv2.VideoWriter_fourcc('X', '2', '6', '4')
out = cv2.VideoWriter('output_video.mp4', fourcc,
                      cap.get(cv2.CAP_PROP_FPS), (1280, 720))


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:

        detection.setImage(frame)

        left_fit, right_fit, left_curvature, right_curvature, left_fitx, right_fitx, dif_meters, left_base, right_base = detection.process_detection(
            mtx, dist)
        sanity = True
        if lastLeftLine.detected and lastRightLine.detected:

            lastLeftLine.diffs = lastLeftLine.current_fit - left_fit
            lastLeftLine.line_base_pos = left_base
            lastLeftLine.current_fit = left_fit
            lastLeftLine.radius_of_curvature = left_curvature
            lastLeftLine.recent_xfitted.append(left_fitx)
            if len(lastLeftLine.recent_xfitted) > smooth:
                lastLeftLine.recent_xfitted.pop(0)
            if len(lastLeftLine.recent_xfitted) == smooth:
                lastLeftLine.bestx = get_avarage_xfitted(
                    lastLeftLine.recent_xfitted)
            lastRightLine.diffs = lastRightLine.current_fit - right_fit
            lastRightLine.line_base_pos = right_base
            lastRightLine.current_fit = right_fit
            lastRightLine.radius_of_curvature = right_curvature
            lastRightLine.recent_xfitted.append(right_fitx)
            if len(lastRightLine.recent_xfitted) > smooth:
                lastRightLine.recent_xfitted.pop(0)
            if len(lastRightLine.recent_xfitted) == smooth:
                lastRightLine.bestx = get_avarage_xfitted(
                    lastRightLine.recent_xfitted)
            out.write(detection.draw_final_image(lastLeftLine, lastRightLine))
        else:
            lastLeftLine.detected = True
            lastLeftLine.line_base_pos = left_base
            lastLeftLine.current_fit = left_fit
            lastLeftLine.best_fit = left_fit
            lastLeftLine.bestx = left_fitx
            lastLeftLine.radius_of_curvature = left_curvature
            lastLeftLine.recent_xfitted.append(left_fitx)

            lastRightLine.detected = True
            lastRightLine.line_base_pos = right_base
            lastRightLine.current_fit = right_fit
            lastRightLine.best_fit = right_fit
            lastRightLine.bestx = right_fitx

            lastRightLine.radius_of_curvature = right_curvature
            lastRightLine.recent_xfitted.append(right_fitx)
            out.write(detection.draw_final_image(lastLeftLine, lastRightLine))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()
out.release()
cv2.destroyAllWindows()
