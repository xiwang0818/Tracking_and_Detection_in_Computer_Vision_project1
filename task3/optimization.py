import numpy as np
import cv2 as cv
import os.path as osp
import matplotlib.pyplot as plt
from task1 import get_cam_matrix, Drawer
from task1.pnp import compute_validsift, get_obpts
from task2.pnp_ransac import sift_create, project_points, matching, matched_sift_extract, draw_boxfr


class Tukey:

    def __init__(self, c=4.685):
        self.c = c

    def rho(self, e):
        a = self.c ** 2 / 6 * (1 - (1 - (e / self.c) ** 2) ** 3)
        b = self.c ** 2 / 6
        return np.where(np.abs(e) < self.c, a, b)

    def phi(self, e):
        a = e * np.square(1 - (e / self.c) ** 2)
        return np.where(np.abs(e) < self.c, a, 0)

    def w(self, e):
        a = np.square(1 - (e / self.c) ** 2)
        return np.where(np.abs(e) < self.c, a, 0)


def MAD(e):

    return np.median(np.abs(e))

# IRLS Algorithmus
# Inner variables:
# kp3d_b_m: matched backprojected sift on box in (n-1)th pics in 3d
# kpts2d_m: matched kp in nth image (with original 3d points in n-1) in 2d
# rv_init, tv_init: initialized rvec and tvec from pnp-ransac
# kp_prj_upd: updated projected kp in 2d
def IRLS(kp3d_b_m, kpts2d_m, rv_init, tv_init, iter_para, tau, cameraMatrix):
    tukey = Tukey()
    theta = np.hstack([rv_init.reshape(-1), tv_init.reshape(-1)])
    t = 0
    lmd = 0.001
    u = tau + 1

    while t < iter_para and u > tau:
        rvec = theta[:3].reshape(3, 1)
        tvec = theta[3:].reshape(3, 1)
        # project matched 3d in n-1 into 2d in n
        kp3d_m_prj, jacobian = cv.projectPoints(kp3d_b_m, rvec, tvec, cameraMatrix, distCoeffs=None)
        kp3d_m_prj = kp3d_m_prj.reshape(-1, 2)

        e = np.subtract(kp3d_m_prj, kpts2d_m).reshape(-1)
        sigma = 1.48257968 * MAD(e)
        elem = tukey.w(e / sigma).reshape(-1)

        W = np.diag(elem)
        # E: Energy Function
        E = np.sum(tukey.rho(e))
        J = jacobian[:, :6] / sigma
        delta = -np.linalg.inv(J.T @ W @ J + lmd * np.eye(6)) @ J.T @ W @ (e / sigma)

        # update
        theta_upd = theta + delta
        kp_prj_upd, _ = cv.projectPoints(kp3d_b_m, theta_upd[0:3].reshape(3, 1), theta_upd[3:].reshape(3, 1), cameraMatrix, distCoeffs=None)
        kp_prj_upd = kp_prj_upd.reshape(-1, 2)
        e_upd = np.subtract(kp_prj_upd, kpts2d_m).reshape(-1)
        E_upd = np.sum(tukey.rho(e_upd))

        if E_upd > E:
            lmd = 10 * lmd
        else:
            lmd = 0.1 * lmd
            theta = theta + delta
        u = np.linalg.norm(delta)
        t = t + 1
    return theta[:3].reshape(3, 1), theta[3:].reshape(3, 1)


def main():
    #initialization
    corners = get_obpts()
    cameraMatrix = get_cam_matrix()
    # You may change trajectory here in oder to run in your work place
    img_foler = '/Users/xiwang/Desktop/Praktikum_Tracking_and_Detection_in_CV/data/tracking/'
    data = np.load('init_rtvec.npz')
    rvec = data['rvec']
    tvec = data['tvec']

    rvec_list, tvec_list = [], []
    rvec_list.append(rvec)
    tvec_list.append(tvec)

    for i in range (24):
        # n-1 pic
        img_ = cv.cvtColor(cv.imread(osp.join(img_foler, f'color_0000{6+i:02d}.JPG')), cv.COLOR_BGR2RGB)
        kp2d_box, kp3d_box, des1 = compute_validsift(rvec, tvec, img_)
        #n pic
        img = cv.cvtColor(cv.imread(osp.join(img_foler, f'color_0000{7+i:02d}.JPG')), cv.COLOR_BGR2RGB)
        kpts2d, des2 = sift_create(img)
        query_index, train_index = matching(des1, des2)
        kp3d_b_m, kpts2d_m = matched_sift_extract(train_index, query_index, kp3d_box, kpts2d)

        iter_para = 60
        tau = 0    #update threshold
        rvec, tvec = IRLS(kp3d_b_m, kpts2d_m, rvec, tvec, iter_para, tau, cameraMatrix)
        rvec_list.append(rvec.reshape(3, 1))
        tvec_list.append(tvec.reshape(3, 1))

        vertice2d = project_points(rvec, tvec, cameraMatrix)
        #draw_boxfr(img, vertice2d)

    #draw
    #plt.show()
    drawer = Drawer(scale=0.1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    drawer.draw_cameras(ax, rvec_list, tvec_list, 0.1)
    drawer.draw_box(ax, corners)
    plt.show()


if __name__ == '__main__':
    main()