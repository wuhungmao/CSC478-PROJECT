# set the path
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
DATA = "/virtual/wuhungma/data/"

# ---------------- 1. RGB ----------------
loaded_png = cv2.imread(DATA + "rgb/0000.png")

# print the shape and dtype
print("loaded_png.shape", loaded_png.shape)
print("loaded_png.dtype", loaded_png.dtype)

# ---------------- 2. DEPTH ----------------
# load the tiff file, and convert to mm
loaded_tiff = cv2.imread(DATA + "depth/0000_depth.tiff", cv2.IMREAD_UNCHANGED)
mm = loaded_tiff.astype(np.float32) / 65535 * 100
print("min", mm.min())
print("max", mm.max())
print("median", np.median(mm))

# TODO: count invalid pixels
#   - print the raw dtype of loaded_tiff (expect uint16)

#   - count how many pixels in loaded_tiff equal 0, print it
print("dtype", loaded_tiff.dtype)
print("num of pixels", (loaded_tiff == 0).sum())

# TODO: save a depth picture
#   - import matplotlib.pyplot as plt (top of file)
#   - show mm as an image with a colormap
plt.imshow(mm)
#   - add a colorbar labelled "mm"
plt.colorbar(label="mm")
#   - save it to depth_vis.png, then open that png in VS Code
plt.savefig("/student/wuhungma/CSC478/depth_vis.png")

# ---------------- 3. POSES ----------------
# load pose, and reshape row 0
pose = np.loadtxt(os.path.join(DATA, "pose.txt"), delimiter=",")

print("shape", pose.shape)   # expect (number_of_frames, 16)

# TODO: turn row 0 into a 4x4 matrix
#   - take the first row of pose (16 numbers)

# TODO: how far does the camera move between frames?
#   - build T for frame 0 and frame 1 (same steps as above)
def get_T(pose, i):
    load_row = pose[i]
    #   - reshape it to 4x4, call it T, print it
    load_row = load_row.reshape((4, 4))
    #   - CHECK: is the bottom row [0 0 0 1]?
    #       if not, the numbers were stored column by column -> transpose T
    if load_row[3, 0] != 0 or load_row[3, 1] != 0 or load_row[3, 2] != 0 or load_row[3, 3] != 1:
        print("Bottom row is not [0 0 0 1], transposing T")
        load_row = load_row.T
    # TODO: sanity-check the rotation (Lecture 3)
    #   - R = top-left 3x3 of T
    print("R:\n", load_row[:3, :3])
    #   - print R @ R.T   (should be ~identity)
    print("R @ R.T:\n", load_row[:3, :3] @ load_row[:3, :3].T)
    #   - print det(R)    (should be ~1)
    print("det(R):", np.linalg.det(load_row[:3, :3]))
    return load_row
#   - camera position = last column, first 3 rows
#   - print both positions, and the distance between them (np.linalg.norm)
#   - units are mm; expect a small number

T0 = get_T(pose, 0)
T1 = get_T(pose, 1)

P0 = T0[:3, 3]
P1 = T1[:3, 3]
print("T0:\n", T0)
print("P1", P0)
print("P2", P1)

print("distance", np.linalg.norm(P2 - P1))

# ---------------- 4. FRAME COUNT ----------------
print("frame count", len(os.listdir(DATA + "rgb")))
