import cv2
from detector.holds.model.yolov4 import Yolov4
import torch

class HoldsDetector(object):
    def __init__(self):
        pass

    def vision_detect_holds(self, image, lower_area=4500, upper_area=5650, draw_contours=True):
        shifted = cv2.pyrMeanShiftFiltering(image, 12, 20)
        canny = cv2.Canny(shifted, 80, 80 * 3)
        processed_img = cv2.GaussianBlur(canny, (5, 5), 0)
        _, processed_img = cv2.threshold(processed_img, 40, 255, cv2.THRESH_BINARY)
        # Find contours
        contours, hierarchy = cv2.findContours(processed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if lower_area < area < upper_area:
                (x, y, w, h) = cv2.boundingRect(c)
                if draw_contours:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        return processed_img

    def ml_detect_holds(self, image):

        n_classes = ''
        weightfile = ''
        imgfile = ''
        namesfile = ''


        model = Yolov4(n_classes=n_classes)

        pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
        model.load_state_dict(pretrained_dict)

        use_cuda = 1
        if use_cuda:
            model.cuda()
