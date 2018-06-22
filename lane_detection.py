import numpy as np
from camera_calibration import undistort_image
from sobel import applySobel
import matplotlib.pyplot as plt
from color_masks import white_yellow_mask
import cv2
from line import Line


class LaneDetection:

    def __init__(self, margin=100, minpix=50, nwindows=9, smooth=5):
        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []
        # Set the width of the windows +/- margin
        self.margin = margin
        # Set minimum number of pixels found to recenter window
        self.minpix = minpix
        # Choose the number of sliding windows
        self.nwindows = nwindows
        self.left_fit = []
        self.right_fit = []
        # Keep track of last n lines fitsx
        self.left_n_fitsx = []
        self.right_n_fitsx = []
        self.left_n_fits = []
        self.right_n_fits = []
        self.left_n_curvature = []
        self.right_n_curvature = []

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700  # meters per pixel in x dimension

        self.smooth = smooth
        self.reset = True
        # count frames
        self.count = 0

    def setImage(self, image):
        self.image = image
        self.imshape = image.shape
        self.count += 1

    def get_undistorted_image(self, mtx, dist):
        if(self.image):
            return undistort_image(self.image, mtx, dist)
        else:
            print('Image not loaded')
            return None

    def warp_transformation(self, maskedImage, undistImage=[]):
        # work on defining perspective transformation area

        d_left_corner = 238
        u_left_corner = 586
        u_right_corner = 695
        d_right_corner = 1070
        src = np.float32([
            [d_left_corner, 690],
            [u_left_corner, 455],
            [u_right_corner, 455],
            [d_right_corner, 690]
        ])

        dst = np.float32([
            [320, 720],
            [320, 0],
            [920, 0],
            [920, 720],
        ])

        if len(undistImage) > 0:
            vertices1 = np.asarray([src], dtype=np.int32)
            poly = cv2.polylines(undistImage, vertices1, 1, (0, 0, 255))
            cv2.imwrite('./output_images/vertices.png', poly)

        M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)
        self.warped = cv2.warpPerspective(
            maskedImage, M, (self.imshape[1], self.imshape[0]), flags=cv2.INTER_NEAREST)

        # Get non zeros pixels from warped image
        self.nonzero = self.warped.nonzero()
        self.nonzeroy = np.array(self.nonzero[0])
        self.nonzerox = np.array(self.nonzero[1])

        # Generate Data for Y space
        self.ploty = np.linspace(
            0, self.warped.shape[0]-1, self.warped.shape[0])

        # Create autput image
        self.out_img = np.dstack((self.warped, self.warped, self.warped))*255

        # Set height of windows
        self.window_height = np.int(self.warped.shape[0]//self.nwindows)

    def get_nonzero_pixels(self):
        # Identify the x and y positions of all nonzero pixels in the image
        return self.nonzero

    def calc_offset(self):
        histogram = np.sum(self.warped[self.warped.shape[0]//2:, :], axis=0)

        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint]) - 82
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint + 150
        self.left_base = midpoint - leftx_base - 82
        self.right_base = rightx_base - (midpoint + 150)

        dif = 1280 - leftx_base - rightx_base

        self.dif_meters = dif * self.xm_per_pix
        if dif < 0:
            self.text_offset = 'Offset: ' + \
                "{:.3f}".format(abs(self.dif_meters)) + ' meters to the left'
        elif dif > 0:
            self.text_offset = 'Offset: ' + \
                "{:.3f}".format(abs(self.dif_meters)) + ' meters to the right'
        else:
            self.text_offset = 'Car at the center of the lane'

        # print(leftx_base, rightx_base, (640 - leftx_base) - (rightx_base - 640))

    def calc_lanes_fits(self):

        histogram = np.sum(self.warped[self.warped.shape[0]//2:, :], axis=0)

        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        print(midpoint, leftx_base, rightx_base)
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.warped.shape[0] - (window+1)*self.window_height
            win_y_high = self.warped.shape[0] - window*self.window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            # Draw the windows on the visualization image
            cv2.rectangle(self.out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(self.out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                              (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                               (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > self.minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = np.int(
                    np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

        # Extract left and right line pixel positions
        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)

        self.reset = False
        print(self.left_fit, self.right_fit)

    def calc_fits_from_previous(self):

        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!

        self.left_lane_inds = ((self.nonzerox > (self.left_fit[0]*(self.nonzeroy**2) + self.left_fit[1]*self.nonzeroy + self.left_fit[2] - self.margin)) & (
            self.nonzerox < (self.left_fit[0]*(self.nonzeroy**2) + self.left_fit[1]*self.nonzeroy + self.left_fit[2] + self.margin)))

        self.right_lane_inds = ((self.nonzerox > (self.right_fit[0]*(self.nonzeroy**2) + self.right_fit[1]*self.nonzeroy + self.right_fit[2] - self.margin)) & (
            self.nonzerox < (self.right_fit[0]*(self.nonzeroy**2) + self.right_fit[1]*self.nonzeroy + self.right_fit[2] + self.margin)))

        # Again, extract left and right line pixel positions
        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty,  self.rightx, 2)

        self.left_n_fits.append(self.left_fit)
        self.right_n_fits.append(self.right_fit)

        if len(self.left_n_fits) > self.smooth:
            self.left_n_fits.pop(0)
            self.right_n_fits.pop(0)

    def calc_fitsx(self):
        self.left_fitx = self.left_fit[0]*self.ploty**2 + \
            self.left_fit[1]*self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0]*self.ploty**2 + \
            self.right_fit[1]*self.ploty + self.right_fit[2]

        self.left_n_fitsx.append(self.left_fitx)
        self.right_n_fitsx.append(self.right_fitx)

        if len(self.left_n_fitsx) > self.smooth:
            self.left_n_fitsx.pop(0)
            self.right_n_fitsx.pop(0)

    def calc_curvature(self):

        y_eval = np.max(self.ploty)

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(
            self.lefty*self.ym_per_pix, self.leftx*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(
            self.righty*self.ym_per_pix, self.rightx*self.xm_per_pix, 2)

        # Calculate the new radii of curvature
        self.left_curverad = (
            (1 + (2*left_fit_cr[0]*y_eval*self.ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        self.right_curverad = (
            (1 + (2*right_fit_cr[0]*y_eval*self.ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        self.curvature = (self.left_curverad + self.right_curverad) / 2
        # Now our radius of curvature is in meters
        # print('left: ', self.left_curverad,
        #   'm Right: ', self.right_curverad, 'm')

    def draw_search(self):

        self.out_img[self.lefty,
                     self.leftx] = [255, 0, 0]
        self.out_img[self.righty,
                     self.rightx] = [0, 0, 255]
        # Create an image to draw on and an image to show the selection window
        window_img = np.zeros_like(self.out_img)
        # Color in left and right line pixels

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array(
            [np.transpose(np.vstack([self.left_fitx-self.margin, self.ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx+self.margin,
                                                                        self.ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([self.right_fitx-self.margin, self.ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([self.right_fitx+self.margin,
                                                                         self.ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(self.out_img, 1, window_img, 0.3, 0)

        # cv2.imshow('Warped', result)

    def draw_final_image(self,  leftLine, rightLine):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array(
            [np.transpose(np.vstack([leftLine.bestx, self.ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([rightLine.bestx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(
            color_warp, self.Minv, (self.undistImage.shape[1], self.undistImage.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(self.undistImage, 1, newwarp, 0.3, 0)
        curvature = (leftLine.radius_of_curvature +
                     rightLine.radius_of_curvature) / 2

        if curvature > 3000:
            text_01 = 'Straight road'
        elif (self.left_fit[1]+self.right_fit[1])/2 > 0:
            text_01 = 'Left curve radius: ' + \
                str(int(curvature)) + ' meters'
        else:
            text_01 = 'Right curve radius: ' + \
                str(int(curvature)) + ' meters'
        offset = (leftLine.line_base_pos -
                  rightLine.line_base_pos) * self.xm_per_pix
        if offset < 0:
            text_offset = 'Offset: ' + \
                "{:.3f}".format(abs(offset)) + ' meters to the left'
        elif offset > 0:
            text_offset = 'Offset: ' + \
                "{:.3f}".format(abs(offset)) + ' meters to the right'
        else:
            text_offset = 'Offset: center'

        cv2.putText(result, text_01, (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))
        cv2.putText(result, text_offset, (50, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))

        cv2.imshow('Final', result)
        return result

    def get_avarage_xfitted(self):
        left_average = 0
        right_average = 0
        if len(self.left_n_fitsx) == self.smooth and len(self.right_n_fitsx) == self.smooth:
            for fitx in self.left_n_fitsx:
                left_average += fitx
            for fitx in self.right_n_fitsx:
                right_average += fitx
            left_average /= 5
            right_average /= 5
            return left_average, right_average
        else:
            print('Fitx Array not completed!')
            exit(-1)

    def sanityCheck(self, leftLine=Line(), rightLine=Line()):
        # Execute a sanity check and return true if it pass and false if fails
        curvature = False
        previous_curvature = True
        distance = False
        parallel = False
        meters_diference = 500

        if (self.left_curverad + self.right_curverad) / 2 > 3000 or abs(self.left_curverad - self.right_curverad) < meters_diference:
            curvature = True

        if abs(self.xm_per_pix*(self.left_fitx[-1] - 82 + 40) - self.xm_per_pix*(self.right_fitx[-1] + 150)) - 2.8 > 0:
            distance = True
        # 0.7 meters to be roughly parallel
        if self.xm_per_pix*abs(abs(self.left_fitx[0] - self.right_fitx[0]) - abs(self.left_fitx[-1] - self.right_fitx[-1])) < 0.7:
            parallel = True
        return curvature & previous_curvature & distance & parallel

    def process_detection(self, mtx, dist):

        self.undistImage = undistort_image(self.image, mtx, dist)

        # cv2.imwrite('./output_images/undist.png', self.undistImage)
        # cv2.imwrite('./output_images/original.png', self.image)

        # exit(-1)
        # Get Edges with sobel
        sobelImage = applySobel(self.undistImage)

        # cv2.imshow('Sobel Mix', sobelImage)

        colorMaskImage = white_yellow_mask(self.undistImage)

        # cv2.imshow('Color Mx', colorMaskImage)

        colorSobelImage = np.zeros_like(colorMaskImage)

        colorSobelImage[(sobelImage == 1) | (colorMaskImage == 1)] = 1

        self.warp_transformation(colorMaskImage)

        if self.reset == False:
            self.calc_fits_from_previous()
        else:
            self.calc_lanes_fits()

        self.calc_fitsx()

        self.calc_curvature()

        self.calc_offset()

        self.draw_search()

        return self.left_fit, self.right_fit, self.left_curverad, self.right_curverad, self.left_fitx, self.right_fitx, self.dif_meters, self.left_base, self.right_base

        # self.draw_final_image(undistImage)
