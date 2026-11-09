# inspect_data.py
# Sanity-check the C3VDv2 data formats before building on them. Findings:
#   RGB   : 1080x1350x3 uint8 (BGR)
#   Depth : uint16 0..65535 -> 0..100 mm; 0 = invalid, 100 = clamped "too far" (also invalid)
#   Pose  : 218 rows x 16 numbers, stored COLUMN-major -> reshape(4,4) then transpose
#           T = [R t; 0 1], t in mm, same world frame as coverage_mesh.obj
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
DATA = "/virtual/wuhungma/data/"


# ---------------- 1. RGB ----------------
# One photo → its shape and dtype (expect 1080x1350x3, uint8)
loaded_png = cv2.imread(DATA + "rgb/0000.png")
print("loaded_png.shape", loaded_png.shape)
print("loaded_png.dtype", loaded_png.dtype)

# ---------------- 2. DEPTH ----------------
# Raw depth TIFF (uint16) → depth in mm (float, 0..100)
loaded_tiff = cv2.imread(DATA + "depth/0000_depth.tiff", cv2.IMREAD_UNCHANGED)
mm = loaded_tiff.astype(np.float32) / 65535 * 100
print("min", mm.min())
print("max", mm.max())
print("median", np.median(mm))

# Raw dtype + number of invalid (0) pixels
print("dtype", loaded_tiff.dtype)
print("num of pixels", (loaded_tiff == 0).sum())

# Depth in mm → coloured picture with a mm colorbar (depth_vis.png)
plt.imshow(mm)
plt.colorbar(label="mm")
plt.savefig("/student/wuhungma/CSC478/depth_vis.png")

# ---------------- 3. POSES ----------------
# pose.txt → (218, 16) array, one row per frame
pose = np.loadtxt(os.path.join(DATA, "pose.txt"), delimiter=",")
print("shape", pose.shape)   # expect (number_of_frames, 16)

# get_T: row i of pose.txt → 4x4 camera pose T (transposed, since the file is column-major)
#        also prints checks that R is a valid rotation: R @ R.T ≈ I, det(R) ≈ 1
def get_T(pose, i):
    load_row = pose[i]
    load_row = load_row.reshape((4, 4))
    if load_row[3, 0] != 0 or load_row[3, 1] != 0 or load_row[3, 2] != 0 or load_row[3, 3] != 1:
        print("Bottom row is not [0 0 0 1], transposing T")
        load_row = load_row.T
    print("R:\n", load_row[:3, :3])
    print("R @ R.T:\n", load_row[:3, :3] @ load_row[:3, :3].T)
    print("det(R):", np.linalg.det(load_row[:3, :3]))
    return load_row

# Poses of frame 0 and 1 → camera positions t → how far the camera moved (mm)
T0 = get_T(pose, 0)
T1 = get_T(pose, 1)

P0 = T0[:3, 3]
P1 = T1[:3, 3]
print("T0:\n", T0)
print("P0", P0)
print("P1", P1)

print("distance", np.linalg.norm(P1 - P0))

# ---------------- 4. FRAME COUNT ----------------
# rgb/ folder → number of frames (218)
print("frame count", len(os.listdir(DATA + "rgb")))
