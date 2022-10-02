import os
import cv2
import numpy as np
import json

""" Reference: 
        * https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
        * https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
"""

class FishEyeCameraModel():
    def __init__(self):
        self.K = None
        self.D = None
        self.DIM = None

    def build(self, img_folder, checkerboard_size):
        CHECKERBOARD = checkerboard_size
        subpix_criteria = (
            cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 
            30, 
            0.1)
        calibration_flags = \
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+ \
            cv2.fisheye.CALIB_CHECK_COND+ \
            cv2.fisheye.CALIB_FIX_SKEW
        
        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        _img_shape = None
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = [os.path.join(img_folder, i) for i in sorted(os.listdir(img_folder))]

        for fname in images:
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(
                gray, 
                CHECKERBOARD, 
                cv2.CALIB_CB_ADAPTIVE_THRESH+\
                cv2.CALIB_CB_FAST_CHECK+\
                cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)     

        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = \
            cv2.fisheye.calibrate(
                objpoints,
                imgpoints,
                gray.shape[::-1],
                K,
                D,
                rvecs,
                tvecs,
                calibration_flags,
                (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(_img_shape[::-1]))
        print("K=np.array(" + str(K.tolist()) + ")")
        print("D=np.array(" + str(D.tolist()) + ")")

        self.DIM = (_img_shape[1], _img_shape[0])
        self.K =  K
        self.D =  D

    def save(self, path):
        model = {
            "K": self.K.tolist(),
            "D": self.D.tolist(),
            "DIM": self.DIM
        }
        fp = open(path, "w")
        json.dump(model, fp)
        fp.close()

    def load(self, path):
        fp = open(path, "r")
        model = json.load(fp)
        fp.close()
        self.K = np.array(model['K'])
        self.D = np.array(model['D'])
        self.DIM = np.array(model['DIM'])

    def undistort(self, img):
        if type(img) == str:
            img = cv2.imread(img)
        
        h,w = img.shape[:2]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, 
            self.D, 
            np.eye(3), 
            self.K, 
            self.DIM, 
            cv2.CV_16SC2)
        undistorted_img = cv2.remap(
            img, map1, map2, 
            interpolation=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

if __name__ == "__main__":
    CHECKERBOARD_SIZE = (6, 9)
    MODEL_NAME = "undistortion/camera.json"
    CHECKERIMGDIR = "undistortion/checkerboard"
    TESTPATH = "undistortion/test.jpeg"

    c = FishEyeCameraModel()
    if not os.path.isfile(MODEL_NAME):
        c.build(CHECKERIMGDIR, CHECKERBOARD_SIZE)
        c.save(MODEL_NAME)
    else:
        c.load(MODEL_NAME)

    # test model
    undistorted_img = c.undistort(TESTPATH)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
