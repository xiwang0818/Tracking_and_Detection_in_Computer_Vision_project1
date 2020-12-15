import numpy as np
import cv2 as cv
import os.path as osp
import matplotlib.pyplot as plt
from task1 import get_cam_matrix, get_obpts


def sift_create(img):
    sift = cv.SIFT_create(edgeThreshold=10, sigma=1.6)
    kpts, des = sift.detectAndCompute(img, None)
    kpts2d = cv.KeyPoint_convert(kpts) # kpts2d is an array
    return kpts2d, des

# Brute-force matcher
def matching(des1, des2):
    bf = cv.BFMatcher_create(normType=cv.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    query_index, train_index = [], []
    for match in matches:
        query_index.append(match.queryIdx)
        train_index.append(match.trainIdx)

    return query_index, train_index

# inner variables:
# for following varaibles: 1 represents query image, 2 represents train image
def matched_sift_extract(train_index, query_index, kpts3d_1, kpts2d_2):
    kpts2d_2_m = kpts2d_2[train_index]
    kpts3d_1_m = kpts3d_1[query_index]
    kpts2d_2_m = np.array(kpts2d_2_m)
    kpts3d_1_m = np.array(kpts3d_1_m)

    return kpts3d_1_m, kpts2d_2_m


def PnPRansac(kpts3d_1_m, kpts2d_2_m, cameraMatrix):
    retval, rvec, tvec, inliers = cv.solvePnPRansac(kpts3d_1_m[:], kpts2d_2_m[:], cameraMatrix, distCoeffs=None,
                                                    useExtrinsicGuess=True, flags=cv.SOLVEPNP_P3P,
                                                    reprojectionError=20, iterationsCount=500)
    if not retval:
        print('Error')
        return None

    return rvec, tvec

# To project points in 3d into 2d
def project_points(rvec, tvec, cameraMatrix):

    obpts = get_obpts()
    vertice2d, jacobian = cv.projectPoints(obpts, rvec, tvec, cameraMatrix, distCoeffs=None)
    vertice2d = np.array(vertice2d).reshape((8, 2))
    # np.savez('../task3/init_kpts3d_0_m', kpts3d_0_m=kpts3d_0_m)
    return vertice2d

# To draw box frame
def draw_boxfr(img, vertice2d):

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter(vertice2d[:, 0], vertice2d[:, 1], s=20, c='b')
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [1, 5], [2, 6], [3, 7], [0, 4], [4, 5], [5, 6], [6, 7], [7, 4]]
    for start, end in edges:
        xs = [vertice2d[start][0], vertice2d[end][0]]
        ys = [vertice2d[start][1], vertice2d[end][1]]
        ax.plot(xs, ys, color='b')

# Inner variables:
# kpts3d_1: all detected valid kp of the model
# des1: corresponding descriptors
# in variable name, '1' indicates query model and '2' indicates train pic; m indicates 'matched'
def main():
    cameraMatrix = get_cam_matrix()
    # You may change trajectory here in oder to run in your work place
    old_data = np.load('/Users/xiwang/Desktop/Praktikum_Tracking_and_Detection_in_CV/task1b/saved_kpts.npz')
    kpts3d_1 = old_data['kpts3d_all']
    des1 = old_data['descriptors_all']
    # You may change trajectory here in oder to run in your work place
    img_foler = '/Users/xiwang/Desktop/Praktikum_Tracking_and_Detection_in_CV/data/detection/'
    for i in range(24):
        img = cv.cvtColor(cv.imread(osp.join(img_foler, f'DSC_97{51 + i}.JPG')), cv.COLOR_BGR2RGB)
        kpts2d, des = sift_create(img)
        des2 = des
        kpts2d_2 = kpts2d
        query_index, train_index = matching(des1, des2)
        kpts3d_1_m, kpts2d_2_m = matched_sift_extract(train_index, query_index, kpts3d_1, kpts2d_2)
        rvec, tvec = PnPRansac(kpts3d_1_m, kpts2d_2_m, cameraMatrix)
        vertice2d = project_points(rvec, tvec, cameraMatrix)

        draw_boxfr(img, vertice2d)
    plt.show()


if __name__ == '__main__':
    main()