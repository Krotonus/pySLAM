import cv2
import numpy as np
from skimage.measure import ransac 
from skimage.transform import EssentialMatrixTransform
from skimage.transform import FundamentalMatrixTransform

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def extractRt(E):
    W = np.mat([[0, -1, 0],
                [1, 0, 0 ],
                [0, 0, 1 ]], dtype = float)
    U, d, Vt = np.linalg.svd(E)
    assert np.linalg.det(U) > 0
    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    Rt = np.concatenate([R, t.reshape(3, 1)], axis = 1)
    return Rt

class FeatureExtractor(object):
    def __init__(self, K):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]

    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        #print(ret)
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8),
                maxCorners = 3000,
                qualityLevel = 0.01,
                minDistance = 3)

        kps = [cv2.KeyPoint(f[0][0], f[0][1], _size = 20) for f in feats]
        kps, des = self.orb.compute(img, kps)
        Rt = None
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k = 2)
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        if len(ret) > 0:
            ret = np.array(ret)
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])
            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    EssentialMatrixTransform,
                                    #FundamentalMatrixTransform,
                                    min_samples = 8,
                                    residual_threshold = 0.5,
                                    max_trials = 100)
            ret = ret[inliers]
            Rt = extractRt(model.params)

        self.last = {'kps': kps,
                     'des': des}
        return ret, Rt
