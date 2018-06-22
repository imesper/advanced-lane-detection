import numpy as np
import cv2


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(
            cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(
            cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1
    # Return the result

    binary_output = np.asarray(binary_output, np.float)

    return binary_output


def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # Create a binary image of ones where threshold is met, zeros otherwise
    # ret, binary_output = cv2.threshold(
    # gradmag, mag_thresh[0], mag_thresh[1], cv2.THRESH_BINARY)
    # print(ret, binary_output)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    # Return the binary image
    binary_output = np.asarray(binary_output, np.float)
    return binary_output


def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    sobel_kernel = 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    binary_output = np.zeros_like(absgraddir)

    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    binary_output = np.asarray(binary_output, np.float)
    # Return the binary image
    return binary_output


def applySobel(image):

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(
        image, orient='x', sobel_kernel=5, thresh=(30, 100))
    grady = abs_sobel_thresh(
        image, orient='y', sobel_kernel=5, thresh=(30, 100))
    mag_binary = mag_thresh(
        image, sobel_kernel=3, mag_thresh=(30, 100))
    dir_binary = dir_threshold(
        image, sobel_kernel=3, thresh=(0.7, 1.3))
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1))
             | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined
