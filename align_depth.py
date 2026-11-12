"""
align_depth.py — turn relative inverse depth (C1 / C2) into metric depth (mm)
with ONE scale s and shift t per video:   1/GT ≈ s * pred + t

Usage idea:  python align_depth.py --cond c1   (or c2)

PSEUDOCODE — replace each step with real code.
"""

# STEP 0 — Settings
#   paths: GT depth folder, C1 .npy folder, C2 .npz file, output folder
#   pick condition (c1 / c2) from command-line args
#   N_SAMPLES = 1_000_000, MIN_MM = 1, MAX_MM = 150
path = "/virtual/wuhungma/stereo_c3vd/depth/"
C1_PATH = "/virtual/wuhungma/depth_c1/"
C2_images_depths = "/virtual/wuhungma/depth_c2/images_depths.npz"
import numpy as np
import os
import cv2

N_SAMPLES = 1_000_000
MIN_MM = 1
MAX_MM = 100
COND = "c1"  # or "c2"
# STEP 1 — Load predictions  -> pred: (N, H, W) float32
#   c1: stack sorted depth_c1/NNNN.npy
#   c2: np.load(depth_c2/images_depths.npz)['depths']
if COND == "c1":
    files = sorted(f for f in os.listdir(C1_PATH) if f.endswith('.npy'))
    pred = np.stack([np.load(os.path.join(C1_PATH, f)) for f in files])
else:
    pred = np.load(C2_images_depths)['depths']

# STEP 2 — Load GT  -> gt: (N, H, W) float, in mm
#   cv2.imread(stereo_c3vd/depth/NNNN.png, -1) for each frame (same sorted order!)
#   assert pred.shape == gt.shape
gt = [cv2.imread(os.path.join(path, f), -1) for f in sorted(os.listdir(path)) if f.endswith('.png')]
gt = np.array(gt)
assert pred.shape == gt.shape

# STEP 3 — Valid pixels + sample
#   mask = gt > 0
#   x = pred[mask], y = 1 / gt[mask]
#   randomly pick N_SAMPLES of (x, y)  (fixed seed, so runs repeat exactly)
mask = gt > 0

x = pred[mask]
y = 1 / gt[mask]
np.random.seed(42)
if len(x) > N_SAMPLES:
    indices = np.random.choice(len(x), N_SAMPLES, replace=False)
    x = x[indices]
    y = y[indices]

# STEP 4 — Fit s, t (least squares)
#   A = [x, 1]  (two columns),  solve A @ [s, t] ≈ y  with np.linalg.lstsq
#   print s, t
#   (later, optional: RANSAC if outliers pull the fit)
A = np.vstack([x, np.ones_like(x)]).T
s, t = np.linalg.lstsq(A, y, rcond=None)[0]
print(f"Fitted scale s: {s}, shift t: {t}")

# STEP 5 — Convert all frames to mm
#   inv = s * pred + t
#   depth_mm = 1 / inv  where inv > 0, else invalid
#   clip to [MIN_MM, MAX_MM]
inv = s * pred + t
depth_mm = np.where(inv > 0, np.clip(1 / inv, MIN_MM, MAX_MM), 0)

# STEP 6 — Evaluate
#   overall RMSE (mm) on mask
#   per-frame RMSE -> list of N numbers; print mean / max
#   save per-frame RMSE (.npy or .csv) for the C1-vs-C2 plot later
RMSE = np.sqrt(np.mean((depth_mm[mask] - gt[mask]) ** 2))
print(f"Overall RMSE (mm): {RMSE}")
per_frame_rmse = [np.sqrt(np.mean((depth_mm[i][mask[i]] - gt[i][mask[i]]) ** 2)) for i in range(len(gt))]
print(f"Per-frame RMSE (mm) - Mean: {np.mean(per_frame_rmse)}, Max: {np.max(per_frame_rmse)}")

# STEP 7 — Save   (do this only after STEP 6 numbers look sane, ~a few mm)
#   round depth_mm -> uint16, write NNNN.png into output folder (same format as GT)
uint16_depth_mm = np.round(depth_mm).astype(np.uint16)
output_folder = f"/virtual/wuhungma/depth_{COND}_aligned/"
os.makedirs(output_folder, exist_ok=True)
for i, depth in enumerate(uint16_depth_mm):
    cv2.imwrite(os.path.join(output_folder, f"{i:04d}.png"), depth)