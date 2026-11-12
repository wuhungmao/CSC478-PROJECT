# c3vd2endonerf.py
# Convert our undistorted C3VDv2 video into the folder format Deform3DGS already reads
# (the "EndoNeRF / StereoMIS" format), so we don't have to change Deform3DGS's code.
#
# Input : /virtual/wuhungma/undistorted/rgb/NNNN.png, depth/NNNN_depth.tiff, K.txt
#         /virtual/wuhungma/data/pose.txt
# Output: /virtual/wuhungma/stereo_c3vd/            <- name MUST contain "stereo_"
#             images/NNNN.png    640x512 colour
#             depth/NNNN.png     640x512, depth in mm, 16-bit PNG
#             masks/NNNN.png     640x512, 255 = valid pixel, 0 = ignore
#             poses_bounds.npy   (218, 17)
#
# Why "stereo_": the loader (scene/endo_loader.py) then uses its StereoMIS branch, which
#   does NOT flip pose axes. Our poses are already OpenCV style (x right, y down, z forward),
#   proven by check_backproject.py.
# Why 640x512: the loader hard-codes this size. Same shape ratio as 1350x1080 (1.25), so
#   a plain resize works; f just shrinks by the same factor.

# ---------------- 0. SETUP ----------------
# imports: cv2, numpy, os
# UND  = "/virtual/wuhungma/undistorted/"
# DATA = "/virtual/wuhungma/data/"
# OUT  = "/virtual/wuhungma/stereo_c3vd/"
# W_NEW, H_NEW = 640, 512
# create OUT/images, OUT/depth, OUT/masks  (os.makedirs(..., exist_ok=True) -- imwrite fails silently otherwise!)
import cv2
import numpy as np
import os
from inspect_data import get_T
UNDISTORTED = "/virtual/wuhungma/undistorted/"
DATA = "/virtual/wuhungma/data/"
OUT = "/virtual/wuhungma/stereo_c3vd/"
W_NEW, H_NEW = 640, 512
os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT, "depth"), exist_ok=True)
os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)

# ---------------- 1. LOAD K AND POSES ----------------
# K = loadtxt(UND + "K.txt")            -> f = K[0,0]
# scale = W_NEW / 1350                   (≈ 0.474)
# f_new = f * scale                      (≈ 237)  -- the loader computes the centre itself (320, 256)
# poses = loadtxt(DATA + "pose.txt", delimiter=",")
# get_T: copy from inspect_data.py but WITHOUT the prints (218 frames would print a lot)
K = np.loadtxt(os.path.join(UNDISTORTED, "K.txt"))
scale = W_NEW / 1350
f_new = K[0, 0] * scale
poses = np.loadtxt(os.path.join(DATA, "pose.txt"), delimiter=",")

# ---------------- 2. FOR EACH FRAME: RESIZE + SAVE ----------------
# poses_bounds = empty list
# for i, name in enumerate(sorted(os.listdir(UND + "rgb"))):      # name = "0000.png", ...
#   2a. RGB: read, cv2.resize to (W_NEW, H_NEW) with cv2.INTER_AREA (good for shrinking),
#       save to OUT/images/name
#   2b. depth: read the tiff (IMREAD_UNCHANGED), resize with cv2.INTER_NEAREST (never blend depths),
#       convert to mm (same formula as before), set invalid pixels (<= 0 or >= 100) to 0,
#       round to whole mm and convert to uint16, save to OUT/depth/name   (.png! the loader only finds *.png)
#       NOTE: whole mm is a first, simple version. If depth looks "stair-stepped" later, we'll improve it.
#   2c. mask: 255 where depth is valid (> 0), else 0, as uint8; save to OUT/masks/name
#   2d. pose row for poses_bounds.npy:
#         T = get_T(poses, i)                      4x4 camera -> world, in mm
#         P = T[:3, :4]                            3x4  [R | t]
#         hwf = column [H_NEW, W_NEW, f_new]       3x1  (the loader reads focal from here)
#         row = [P | hwf] -> 3x5 -> flatten to 15 numbers, then append [near, far] = [1, 100]  -> 17 numbers
#       append row to poses_bounds
#   (optional) print i every 50 frames so you see progress
poses_bounds = []
for i, name in enumerate(sorted(os.listdir(UNDISTORTED + "rgb"))):
    rgb = cv2.imread(os.path.join(UNDISTORTED, "rgb", name))
    rgb_resized = cv2.resize(rgb, (W_NEW, H_NEW), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, "images", name), rgb_resized)
    depth = cv2.imread(os.path.join(UNDISTORTED, "depth", name.replace(".png", "_depth.tiff")), cv2.IMREAD_UNCHANGED)
    depth_resized = cv2.resize(depth, (W_NEW, H_NEW), interpolation=cv2.INTER_NEAREST)
    depth_mm = (depth_resized.astype(np.float32) / 65535 * 100).round().astype(np.uint16)
    depth_mm[(depth_mm <= 0) | (depth_mm >= 100)] = 0
    cv2.imwrite(os.path.join(OUT, "depth", name.replace(".png", ".png")), depth_mm)
    mask = np.where(depth_mm > 0, 255, 0).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, "masks", name), mask)

    T = get_T(poses, i)
    P = T[:3, :4]
    hwf = np.array([[H_NEW], [W_NEW], [f_new]])
    row = np.hstack((P, hwf)).flatten()
    near_far = np.array([1, 100])
    row = np.hstack((row, near_far))
    poses_bounds.append(row)

    if i % 50 == 0:
        print(f"Processed frame {i}")
# ---------------- 3. SAVE POSES ----------------
# np.save(OUT + "poses_bounds.npy", np.array(poses_bounds))     -> shape (218, 17)
np.save(os.path.join(OUT, "poses_bounds.npy"), np.array(poses_bounds))

# ---------------- 4. QUICK CHECKS ----------------
# print shape of poses_bounds (expect (218, 17))
# count files in images/, depth/, masks/ (expect 218 each)
# read back OUT/depth/0000.png with IMREAD_UNCHANGED: print dtype (uint16) and median of non-zero values (~40 mm)
print("poses_bounds shape:", np.array(poses_bounds).shape)
print("Number of images:", len(os.listdir(os.path.join(OUT, "images"))))
print("Number of depth maps:", len(os.listdir(os.path.join(OUT, "depth"))))
print("Number of masks:", len(os.listdir(os.path.join(OUT, "masks"))))
depth_check = cv2.imread(os.path.join(OUT, "depth", "0000.png"), cv2.IMREAD_UNCHANGED)
print("Depth check dtype:", depth_check.dtype)
non_zero_depths = depth_check[depth_check > 0]
if non_zero_depths.size > 0:
    print("Median of non-zero depths:", np.median(non_zero_depths)) 