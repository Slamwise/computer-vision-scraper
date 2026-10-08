import os
import cv2
heights = []
for image in os.listdir(r'/home/user/src/darknet/proj/eb/test1'):
    img = cv2.imread(r'/home/user/src/darknet/proj/eb/test1/' + image)
    height, width, channels = img.shape
    heights.append(height)
hmin = min(heights)
for image in os.listdir(r'/home/user/src/darknet/proj/eb/test1'):
    img = cv2.imread(r'/home/user/src/darknet/proj/eb/test1/' + image)
    dim = (800, hmin)
    resize = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(r'/home/user/src/darknet/proj/eb/test1/' + image, resize)