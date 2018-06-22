import cv2
import numpy as np


def white_mask(original):
    """
    Create a mask from the whitish pixels of the frame
    """
    # specify the range of colours that you want to include, you can play with the borders here
    lower_white = (190, 100, 100)
    upper_white = (255, 255, 255)

    white = cv2.inRange(original, lower_white, upper_white)

    mask = np.zeros_like(white)

    mask[white > 0] = 1

    mask = np.asarray(mask, np.float)

    return mask


def yellow_mask_rgb(original):
    """
    Create a mask from the yellowish pixels of the frame
    """
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    lower_yellow = (230, 120, 0)
    upper_yellow = (255, 255, 180)

    yellow = cv2.inRange(original, lower_yellow, upper_yellow)

    # cv2.imshow('Yellow', yellow)

    mask = np.zeros_like(yellow)

    mask[yellow > 0] = 1
    mask = np.asarray(mask, np.float)

    return mask


def yellow_mask_hsv(original):
    """
    Create a mask from the yellowish pixels of the frame
    """

    HSV = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    lower_HSV = (00, 90, 100)
    upper_HSV = (80, 255, 255)

    yellow_HSV = cv2.inRange(HSV, lower_HSV, upper_HSV)

    mask = np.zeros_like(yellow_HSV)

    mask[yellow_HSV > 0] = 1

    mask = np.asarray(mask, np.float)

    return mask


def yellow_mask(original):
    """
    Create a mask from the yellowish pixels of the frame, combining RGB mask and HSV mask
    """

    hsv = yellow_mask_hsv(original)
    rgb = yellow_mask_rgb(original)

    mask = np.zeros_like(rgb)
    mask[(hsv == 1) | (rgb == 1)] = 1

    return mask


def white_yellow_mask(original):

    yellow = yellow_mask(original)
    white = white_mask(original)

    mask = np.zeros_like(white)
    mask[(white == 1) | (yellow == 1)] = 1

    return mask
