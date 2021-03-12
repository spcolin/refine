"""
test for the format of .mat file
"""

import scipy.io as scio
import cv2

mat_path = "E:/refine/datasets/NYUDV2/train/train.mat"

data = scio.loadmat(mat_path)

# print(data.keys())
# print(data["__header__"])
# print(data["__version__"])
# print(data["__globals__"])


# print(len(data['depths']))
# print(len(data['rgbs']))
# print(len(data['raw_depths']))
# print(len(data['raw_depth_filenames']))
# print(len(data['raw_rgb_filenames']))


depth = data['depths'][10]
print(depth.shape)
print(depth[200,200])
# depth = depth.transpose((0, 2, 1)) / 255.0
# print(depth.shape)
# print(depth[2])
# print(depth[1][5])
# print(depth[2][5])

# cv2.imshow("rgb", depth)
# cv2.waitKey(0)