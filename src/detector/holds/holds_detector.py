import cv2
import numpy as np


class HoldsDetector(object):
    def __init__(self):
        pass

    def auto_canny(self, image, sigma=0.5):
        # compute the median of the single channel pixel intensities
        v = np.median(image)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)

        # return the edged image
        return edged

    def detect_holds(self, image):
        shifted = cv2.pyrMeanShiftFiltering(image, 12, 20)
        canny = self.auto_canny(shifted)

        return canny
