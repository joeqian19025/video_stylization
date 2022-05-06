import pickle
import json
import cv2
import numpy as np
import base64

def enpack(im):
    success, im_numpy = cv2.imencode('.jpg', im)
    imdata = im_numpy.tostring()
    return imdata

def depack(imdata):
    im_numpy = np.fromstring(imdata, np.uint8)
    im = cv2.imdecode(im_numpy, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
    return im

class ReshapeTool():
    def __init__(self):
        self.record_H = 0
        self.record_W = 0

    def process(self, img):
        H, W, C = img.shape

        if self.record_H == 0 and self.record_W == 0:
            new_H = H + 128
            if new_H % 64 != 0:
                new_H += 64 - new_H % 64

            new_W = W + 128
            if new_W % 64 != 0:
                new_W += 64 - new_W % 64

            self.record_H = new_H
            self.record_W = new_W

        new_img = cv2.copyMakeBorder(img, 64, self.record_H-64-H,
                                          64, self.record_W-64-W, cv2.BORDER_REFLECT)
        return new_img
