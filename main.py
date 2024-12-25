# imports
import cv2 # opencv for image processing
import numpy as np
from matplotlib import pyplot as plt
import imutils
import pytesseract # pytesseract for OCR
import re # regex for parsing text
# set params
frameWidth = 504
frameHeight = 672

# read test image and process
img = cv2.imread('assets/IMG.jpeg', cv2.IMREAD_COLOR)       # read in image
img = cv2.resize(img, (frameWidth, frameHeight))        # 1/12 scale
#cv2.imshow("initial img", img)      # plot

# increase contrast using histogram equalization
#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      # convert colorspace from BGR to Grayscale
img_hist_equalized = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
channels = cv2.split(img_hist_equalized)      # split channels
channels = list(channels)       # convert from tuple to list
channels[0] = cv2.equalizeHist(channels[0])         # equalize channels
img_hist_equalized = cv2.merge(channels)        # merge channels back
#cv2.imshow("hist equalized", img_hist_equalized) # plot

# Laplacian of Gaussian
img_blur = cv2.GaussianBlur(img_hist_equalized, (3,3), 0)      # apply Gaussian blur
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)       # convert colorspace from BGR to Grayscale
img_laplacian = cv2.Laplacian(img_gray, cv2.CV_8U, 3, 3, 2)     # apply Lapacian 
img_LoG = cv2.convertScaleAbs(img_laplacian)
#cv2.imshow("convert scale abs", img_LoG)        # plot

# Binarization
binarized_thresh = cv2.threshold(img_LoG, 60, 255, cv2.THRESH_BINARY)[1] # binarized threshold 60
#cv2.imshow("binarized", binarized_thresh)       # plot

# Canny edge detection
edges = cv2.Canny(binarized_thresh,170,200)
cv2.imshow("edges", edges)

# Find and remove all connected components (white blobs)
nb_blobs, img_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(binarized_thresh)     # img where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
# The background pixels have value 0.
sizes = stats[:, cv2.CC_STAT_AREA]      # get size of blobs
min_size = 7      # minimum size of blobs 30, NEED TO DECREASE
img_result = np.zeros_like(img_with_separated_blobs)      # empty output img with only biggest components
for index_blob in range(1, nb_blobs):       # filter all components for min size
    if sizes[index_blob] >= min_size:
        img_result[img_with_separated_blobs == index_blob] = 255
img_normalised = cv2.normalize(img_result, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)      # normalise image 
cv2.imshow("normalised", img_normalised)       # print

# Contours
#contours = cv2.findContours(img_normalised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(img_normalised, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)        # contour
for i, contour in enumerate(contours): # i stores iteration number
   if i == 0:
       continue # first contour stores whole image as shape

   epsilon = 0.01 * cv2.arcLength(contour, True) # precision of approximation, True is for closed shape
   approx = cv2.approxPolyDP(contour, epsilon, True) # approximates closed shape

   area = cv2.contourArea(contour) # calculate area of contour
   if area >7: # filter for contour area size
       cv2.drawContours(img, contour, 0, (255, 255, 0), 4) # draw contours
       x, y, w, h = cv2.boundingRect(approx) # returns co-ordinates of contour
       x_mid = int(x + w/3) # approx centre of contour to write on
       y_mid = int(y + h/1.5) # approx centre of contour to write on
       coords = (x_mid, y_mid) # store centre co-ordinates
       colour = (0,0,0) # set text colour
       font = cv2.FONT_HERSHEY_DUPLEX # set text font
       cv2.putText(img, "P", coords, font, 1, colour, 1) # write text

#cv2.imshow("final", img) # plot

# Close cv2
cv2.waitKey(0) # wait for key input
cv2.destroyAllWindows() # close

# Perform OCR
text = pytesseract.image_to_string(img)
print("Extracted Text:")
print(text)