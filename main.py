# imports
import cv2 # opencv for image processing
import pytesseract # pytesseract for OCR
import re # regex for parsing text

# read test image
img = cv2.imread('assets/IMG_5911.jpeg', cv2.IMREAD_COLOR)
print(img.shape)
imgS = cv2.resize(img, (960, 540)) 
cv2.imshow("parking image", imgS)
cv2.waitKey(0)
cv2.destroyAllWindows()



# PIPELINE: 
# 1. Pre-process image
#   - Convert to grayscale
#   - Use adaptive thresholding/edge detection
#   - Remove noise
#   - Complex Layouts: For signs with multiple sections, segment the image into regions using OpenCV's contour detection or template matching.
#   - Low-quality Images: Use image enhancement techniques like histogram equalization or super-resolution.
# 2. Perform OCR
# 3. Parse text using regex
# 