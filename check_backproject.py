# check_backproject.py
# Sanity check: turn a few UNDISTORTED GT depth frames into 3D points in the world,
# then measure how far those points are from coverage_mesh.obj.
# If the median distance is < ~1 mm, the undistortion, depth units AND pose convention are all correct.
#
# Inputs : /virtual/wuhungma/undistorted/depth/NNNN_depth.tiff, /virtual/wuhungma/undistorted/K.txt
#          /virtual/wuhungma/data/pose.txt, /virtual/wuhungma/data/coverage_mesh.obj
# Output : printed distances, and a .ply point cloud you can open later

# ---------------- 0. SETUP ----------------
# Paths, the 5 test frames, and STEP = use every 4th pixel (keeps it fast)
import cv2
import numpy as np
import os
import open3d as o3d
from inspect_data import get_T

DATA = "/virtual/wuhungma/data/"
UND_FILE  = "/virtual/wuhungma/undistorted/"
FRAMES = [0, 50, 100, 150, 200]
STEP   = 4

# ---------------- 1. LOAD K AND POSES ----------------
# K.txt → focal length f and image centre (cx, cy) of the pinhole camera
# pose.txt → (218, 16) array; get_T turns row i into the 4x4 pose
K = np.loadtxt(UND_FILE + "K.txt")
f = K[0, 0]
cx = K[0, 2]
cy = K[1, 2]
poses = np.loadtxt(DATA + "pose.txt", delimiter=",")

# ---------------- 2. PIXEL GRID (done once) ----------------
# Image size → (col, row) coordinates of every STEP-th pixel, each shape (270, 338)
x_vals = np.arange(0, 1350, STEP)
y_vals = np.arange(0, 1080, STEP)
col, row = np.meshgrid(x_vals, y_vals)

# ---------------- 3. FOR EACH FRAME: DEPTH -> 3D POINTS ----------------
# Depth frame → 3D points relative to that frame's camera (p_cam, step 3e)
# That frame's pose (R, t) → the same points in the world coordinate system (p_world, step 3f)
all_points = []
for i in FRAMES:
    # Undistorted depth TIFF → depth in mm at every sampled pixel (Z)
    depth_path = os.path.join(UND_FILE, "depth/%04d_depth.tiff" % i)
    mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 65535 * 100
    Z = mm[row, col]
    # Keep only real measurements (0 = no data, 100 = clamped)
    valid_pixel = (Z > 0) & (Z < 100)
    # Pixel + depth → X, Y, Z in mm relative to the camera (ray from K⁻¹, scaled by Z)
    X = (col - cx) / f * Z
    Y = (row - cy) / f * Z
    X = X[valid_pixel]
    Y = Y[valid_pixel]
    Z = Z[valid_pixel]
    # Three (N,) arrays → one (N, 3) array, one point per row
    three_dim_point = np.stack([X, Y, Z], axis=1)
    # Camera-relative points → world points: rotate by R, then shift by camera position t
    T = get_T(poses, i)
    # Get R and t
    R = T[:3, :3]
    t = T[:3, 3]
    world_coordinates = (R @ three_dim_point.T).T + t
    print(f"Frame {i} gave {world_coordinates.shape[0]} points")
    all_points.append(world_coordinates)

# ---------------- 4. DISTANCE TO THE MESH ----------------
# GT mesh → Open3D raycasting scene (fast nearest-surface lookups)
# All frames' world points (N_total, 3) → distance (mm) from each point to the nearest mesh surface
scene = o3d.t.geometry.RaycastingScene()
mesh = o3d.io.read_triangle_mesh(DATA + "coverage_mesh.obj")
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
points = np.concatenate(all_points, axis=0)
dist = scene.compute_distance(o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)).numpy()
# Distances → median / mean / 90th percentile error (mm)
print("Median distance to mesh:", np.median(dist))
print("Mean distance to mesh:", np.mean(dist))
print("90th percentile distance to mesh:", np.percentile(dist, 90))

# Mesh and points → their bounding boxes (min/max x, y, z), a rough "same region of space?" check
concatenated_points = np.concatenate(all_points, axis=0)
mesh = o3d.io.read_triangle_mesh(DATA + "coverage_mesh.obj")
print("number of vertices in mesh:", len(mesh.vertices))
print("mesh's bounding box:")
print("Min:", mesh.get_min_bound())
print("Max:", mesh.get_max_bound())
print("Points' bounding box:")
print("Min:", concatenated_points.min(axis=0))
print("Max:", concatenated_points.max(axis=0))


# ---------------- 5. RESULT (2026-11-09) ----------------
# 5 frames, ~430k points → median 0.034 mm, mean 0.042 mm, 90th pct 0.089 mm from the mesh
# Bounding boxes match → undistortion, depth units (mm) and pose (camera → world, transposed) are all correct

# ---------------- 6. SAVE ----------------
# World points → check_points.ply (open with coverage_mesh.obj in MeshLab for a report figure)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(concatenated_points)
o3d.io.write_point_cloud(UND_FILE + "check_points.ply", pcd)