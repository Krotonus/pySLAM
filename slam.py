import numpy as np
import cv2
from display import Display
from feature_extractor import FeatureExtractor

W = 1920
H = 1080

screen = Display(W, H)
extractor = FeatureExtractor()


def process_frame(img):
    img = cv2.resize(img, (W//2, H//2))
    matches = extractor.extract(img)

    print(f'{len(matches)} matches')
    '''
    for kp in kps:
        x, y = map(int, kp.pt)
        img = cv2.circle(img, (x, y), radius = 2, color = (0, 255, 0))
    '''
    for p1, p2 in matches:
        u1, v1 = map(lambda x: int(round(x)), p1)
        u2, v2 = map(lambda x: int(round(x)), p2)
        cv2.circle(img, (u1, v1), radius = 3, color = (0, 255, 0))
        cv2.line(img, (u1, v1), (u2, v2), color = (255, 0, 0))

    screen.paint(img)
    
if __name__ == "__main__":
    cap = cv2.VideoCapture("./videos/test.mp4")
    while(cap.isOpened()):
        ret, image = cap.read()
        process_frame(image)
    print("Done.")
