import cv2
import glob
from helpers import grayscale
import numpy as np


def camera_calibration(show_images=False):

    nx = 9
    ny = 6

    objPoints = []
    imgPoints = []
    shape = None
    images = glob.glob('./camera_cal/calibration*.jpg')

    # 3D Coordinates
    objP = np.zeros((nx*ny, 3), np.float32)
    objP[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    for filename in images:
        image = cv2.imread(filename)
        gray = grayscale(image)
        shape = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if show_images == True:
            image = cv2.drawChessboardCorners(
                image, (nx, ny), corners, ret)
            print(filename)
            # cv2.imshow('Cal', image)
            cv2.waitKey()

        if ret == True:
            imgPoints.append(corners)
            objPoints.append(objP)

    # print(objPoints)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objPoints, imgPoints, shape, None, None)

    return mtx, dist


def undistort_image(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def test_calibration():
    mtx, dist = camera_calibration()
    # test Camera Calibration
    testImage = cv2.imread('./camera_cal/calibration2.jpg')
    undistImage = undistort_image(testImage, mtx, dist)
    cv2.imshow("Image", testImage)
    cv2.imshow("Teste Cal", undistImage)
    cv2.imwrite('undistortImage.png', undistImage)
    cv2.waitKey()
