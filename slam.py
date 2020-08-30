import numpy as np
import cv2
from display import Display
from feature_extractor import FeatureExtractor

W = 1920//2
H = 1080//2
F = 1
K = np.array([[F, 0, W//2],
              [0, F, H//2],
              [0, 0, 1]])

screen = Display(W, H)
extractor = FeatureExtractor(K)


def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches, pose = extractor.extract(img)
    if pose is None:
        return

    print(f'{len(matches)} matches')
    print(pose)

    for p1, p2 in matches:
        u1, v1 = extractor.denormalize(p1)
        u2, v2 = extractor.denormalize(p2)
        cv2.circle(img, (u1, v1), radius = 3, color = (0, 255, 0))
        cv2.line(img, (u1, v1), (u2, v2), color = (255, 0, 0))

    screen.paint(img)
    
if __name__ == "__main__":
    cap = cv2.VideoCapture("./videos/test.mp4")
    while(cap.isOpened()):
        ret, image = cap.read()
        process_frame(image)
    print("Done.")
