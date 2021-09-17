import imutils
import cv2 as cv
# import matplotlib.pyplot as plt
from shape_detecter import ShapeDetector

im = cv.imread('hcn.jpg', 1)

resized = imutils.resize(im, width=300)

ratio = im.shape[0] / float(resized.shape[0])

im = cv.cvtColor(im, cv.COLOR_BGR2RGB)

gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

ret, thresh = cv.threshold(gray, 170, 255, 0)

contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

sd = ShapeDetector()

for c in contours:
    M = cv.moments(c)
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape = sd.detect(c)

    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    cv.drawContours(im, [c], -1, (0, 255, 0), 2)
    cv.putText(im, shape, (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # show the output image
    cv.imshow("Image", im)
    cv.waitKey(0)

# cv.drawContours(im, contours, -1, (0, 0, 255), 2)
#
# original = cv.imread('hcn.jpg', 1)
#
# original = cv.cvtColor(original, cv.COLOR_BGR2RGB)
#
# output = [original, thresh]
#
# titles = ['Original', 'Contours']
#
# for i in range(2):
#     plt.subplot(1, 2, i + 1)
#     plt.imshow(output[i])
#     plt.title(titles[i])
#     plt.xticks([])
#     plt.yticks([])
#
# plt.show()
