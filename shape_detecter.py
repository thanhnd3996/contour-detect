import cv2 as cv


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        shape = "unidentified"
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 4:
            (x, y, w, h) = cv.boundingRect(approx)
            ar = w / float(h)

            if ar <= 0.95 or ar >= 1.05:
                shape = "rectangle"

        return shape
