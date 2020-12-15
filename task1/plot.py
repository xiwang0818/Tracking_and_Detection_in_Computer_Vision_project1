import matplotlib.pyplot as plt
import cv2
import numpy as np


class Drawer:

    def __init__(self, scale=0.1):
        self.scale = scale
    # inner variable: loc_array: valid 3d kp
    def draw(self, rvec_list, tvec_list, corners, loc_array):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.draw_box(ax, corners)
        self.draw_cameras(ax, rvec_list, tvec_list, scale=self.scale)
        self.draw_keypoints(ax, loc_array)
        plt.show()

    def draw_cameras(self, ax, rvec_list, tvec_list, scale):
        for rvec, tvec in zip(rvec_list, tvec_list):
            R = cv2.Rodrigues(rvec)[0]
            cam_loc = - (R.T @ tvec).reshape(-1)  # (3,)
            cam_dir = R.T.dot([0, 0, 1]).reshape(-1)  # (3,)
            ax.scatter(xs=cam_loc[0], ys=cam_loc[1], zs=cam_loc[2], zdir='z', s=20, c=None, depthshade=True, color='y')
            camdir_scaled = cam_dir / np.linalg.norm(cam_dir) * scale  # (3,)
            xs = [cam_loc[0], cam_loc[0] + camdir_scaled[0]]
            ys = [cam_loc[1], cam_loc[1] + camdir_scaled[1]]
            zs = [cam_loc[2], cam_loc[2] + camdir_scaled[2]]
            ax.plot(xs=xs, ys=ys, zs=zs, color='g')

    def draw_box(self, ax, corners):
        ax.scatter(0, 0, 0, zdir='z', s=20, c=None, depthshade=True)
        ax.scatter(0.165, 0, 0, zdir='z', s=20, c=None, depthshade=True)
        ax.scatter(0.165, 0.063, 0, zdir='z', s=20, c=None, depthshade=True)
        ax.scatter(0, 0.063, 0, zdir='z', s=20, c=None, depthshade=True)
        ax.scatter(0, 0, 0.093, zdir='z', s=20, c=None, depthshade=True)
        ax.scatter(0.165, 0, 0.093, zdir='z', s=20, c=None, depthshade=True)
        ax.scatter(0.165, 0.063, 0.093, zdir='z', s=20, c=None, depthshade=True)
        ax.scatter(0, 0.063, 0.093, zdir='z', s=20, c=None, depthshade=True)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(0.0, 0.6)
        # draw boxframe
        edges = [[0, 1], [1, 2], [2, 3], [3, 0], [1, 5], [2, 6], [3, 7], [0, 4], [4, 5], [5, 6], [6, 7], [7, 4]]
        for start, end in edges:
            xs = [corners[start][0], corners[end][0]]
            ys = [corners[start][1], corners[end][1]]
            zs = [corners[start][2], corners[end][2]]
            ax.plot(xs=xs, ys=ys, zs=zs, color='black')

    def draw_keypoints(self, ax, loc_array):
        ax.scatter (loc_array[:, 0], loc_array[:, 1], loc_array[:, 2], zdir='z', s=0.1, c=None, depthshade=True, color="r")


