import cv2

class Display(object):
    def __init__(self, W, H):
        self.W = W
        self.H = H

    def paint(self, img):
        cv2.imshow('Video', img)
        cv2.waitKey(10)

