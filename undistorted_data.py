# undistorted_data.py
# Turn fisheye (Scaramuzza / OCamCalib) frames into pinhole frames with a known K.
# Input : DATA/rgb/NNNN.png, DATA/depth/NNNN_depth.tiff, camera_intrinsics.txt
# Output: OUT/rgb/NNNN.png, OUT/depth/NNNN_depth.tiff, OUT/K.txt
#
# The goal: the colonoscope uses a fisheye lens, but 3DGS only understands pinhole cameras (u = f·X/Z + cx, described by K). So you re-create every frame as if a pinhole camera had taken it, and save the K that describes it.


# ---------------- 0. SETUP ----------------
import cv2
import numpy as np
import os

# DATA = folder with rgb/ and depth/
DATA = "/virtual/wuhungma/data/"

# OUT  = new folder for the undistorted data (create OUT/rgb and OUT/depth if missing)
OUT = "/virtual/wuhungma/undistorted/"


# W_in, H_in = 1350, 1080   (original image size)

# ---------------- 1. READ CALIBRATION ----------------
# open camera_intrinsics.txt and print it first -- check what is in it before writing this part
# expected parameters (OCamCalib style):
#   cx, cy            : image centre of the fisheye, in pixels
#   a0, a2, a3, a4    : polynomial f(rho) = a0 + a2*rho^2 + a3*rho^3 + a4*rho^4   (no a1 term)
#   c, d, e           : small affine correction (sensor not perfectly square)
#   maybe an inverse polynomial (world2cam) -- if present, it makes step 3 easier
# store them in variables
# print("camera_intrinsics.txt:", file.readline(os.path.join(DATA, "camera_intrinsics.txt")))
with open(os.path.join(DATA, "camera_intrinsics.txt")) as f:
    for line in f:
        print(line.strip())
width = 1350
height = 1080
cx = 677.739464094188
cy = 543.057997844875
a0 = 767.733695862103
a1 = 0.0
a2 = -0.000592506426558248
a3 = -2.69440266600040e-07
a4 = -2.16380341010063e-10
c = 0.9999
d = 1.10e-4
e = -1.83e-4

# ---------------- 2. CHOOSE THE NEW PINHOLE CAMERA ----------------
# W_out, H_out = output size (start with the same as input)
# f  = focal length in pixels (start ~ 400-500; smaller = wider view, more stretched edges)
# cx_out, cy_out = W_out/2, H_out/2
# K = 3x3 matrix [[f, 0, cx_out],
#                 [0, f, cy_out],
#                 [0, 0, 1     ]]
W_out, H_out = 1350, 1080
f = 500
cx_out, cy_out = W_out / 2, H_out / 2
K = np.array([[f, 0, cx_out],
              [0, f, cy_out],
              [0, 0, 1]])

# ---------------- 3. BUILD THE LOOKUP TABLE (map_x, map_y), done ONCE ----------------
# 3a. make a grid of every output pixel (u, v)          -> hint: np.meshgrid
# 3b. turn each pixel into a 3D ray with K^-1:
#       x = (u - cx_out) / f,   y = (v - cy_out) / f,   z = 1
# 3c. project each ray into the FISHEYE image:
#       r     = sqrt(x^2 + y^2)          (distance of the ray from the optical axis)
#       the fisheye says pixel at radius rho looks along direction [.., .., f(rho)]
#       so we need the rho where  f(rho) / rho  matches  z / r   (same angle)
#       -> if the file has an inverse polynomial: rho = invpoly(angle)
#       -> otherwise: build a table of rho = 0..max, compute angle(rho) for each,
#          then use np.interp to look up rho for every ray's angle
#       WATCH THE SIGN: in OCamCalib f(rho) is often negative (z points the other way). Print a few values.
# 3d. fisheye pixel (before affine):  xd = x / r * rho,   yd = y / r * rho
# 3e. apply affine + centre:
#       map_x = c*xd + d*yd + cx
#       map_y = e*xd + 1*yd + cy
#       (check the order of x/y vs row/col in the file -- OCamCalib sometimes swaps them)
# 3f. convert map_x, map_y to float32  (cv2.remap needs float32)
map_x, map_y = np.meshgrid(np.arange(W_out, dtype=np.float32), np.arange(H_out, dtype=np.float32))
x = (map_x - cx_out) / f
y = (map_y - cy_out) / f
z = 1.0

r = np.maximum(np.sqrt(x**2 + y**2), 1e-8)
  
# Assuming you have the inverse polynomial coefficients from the calibration file
# For example, let's say you have them in a list called inv_poly_coeffs
# You would compute rho using np.polyval on the inverse polynomial coefficients
# For demonstration, let's assume inv_poly_coeffs is defined as follows:
rho_table = np.linspace(0, 800, 2000)  # Example range for rho
f_table    = a0 + a2 * rho_table**2 + a3 * rho_table ** 3 + a4 * rho_table ** 4
tilt_table = rho_table / f_table
rho = np.interp(r, tilt_table, rho_table)
xd = x / r * rho
yd = y / r * rho
# Apply affine transformation and center
map_x = c * xd + d * yd + cx
map_y = e * xd + 1 * yd + cy
map_x = map_x.astype(np.float32)
map_y = map_y.astype(np.float32)

# ---------------- 4. APPLY TO EVERY FRAME ----------------
# for each frame name in sorted(os.listdir(DATA/rgb)):
#   rgb   = read png
#   depth = read tiff with cv2.IMREAD_UNCHANGED  (keep uint16!)
#   rgb_u   = cv2.remap(rgb,   map_x, map_y, cv2.INTER_LINEAR)
#   depth_u = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST)   # nearest: never blend depths
#   (pixels that fall outside the fisheye image -> remap fills them with 0 = invalid, good)
#   save rgb_u and depth_u into OUT/rgb and OUT/depth with the same names
for frame in sorted(os.listdir(DATA + "rgb")):
    rgb = cv2.imread(DATA + "rgb/" + frame)
    depth = cv2.imread(DATA + "depth/" + frame.replace(".png", "_depth.tiff"), cv2.IMREAD_UNCHANGED)
    rgb_u = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR)
    depth_u = cv2.remap(depth, map_x, map_y, cv2.INTER_NEAREST)
    cv2.imwrite(OUT + "rgb/" + frame, rgb_u)
    cv2.imwrite(OUT + "depth/" + frame.replace(".png", "_depth.tiff"), depth_u)

# ---------------- 5. SAVE K ----------------
# np.savetxt(OUT/K.txt, K)
np.savetxt(OUT + "K.txt", K)

# ---------------- 6. QUICK VISUAL CHECK ----------------
# save original frame 0 and undistorted frame 0 side by side into a png
# look for: straight-ish edges, no black holes in the middle, colon still centred
# (the full 3D check against the mesh will be a separate script: check_backproject.py)
rgb_u0 = cv2.imread(OUT + "rgb/0000.png")
rgb0 = cv2.imread(DATA + "rgb/0000.png")
combined = np.hstack((rgb0, rgb_u0))
cv2.imwrite(OUT + "check_0000.png", combined)