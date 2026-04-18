"""
Quick end-to-end smoke test for RegTR after removing MinkowskiEngine.
Run from the repo root:   python test_setup.py
Or from src/:             python ../test_setup.py

Tests:
  1. Grid subsampling unit test (the function we rewrote)
  2. Full forward pass with pretrained weights + real demo data (headless)
"""
import os, sys, urllib.request, zipfile, pathlib, torch
import numpy as np

REPO   = pathlib.Path(__file__).parent.resolve()
SRC    = REPO / "src"
CKPT   = REPO / "trained_models/3dmatch/ckpt/model-best.pth"
SRC_PC = REPO / "data/indoor/test/7-scenes-redkitchen/cloud_bin_0.pth"
TGT_PC = REPO / "data/indoor/test/7-scenes-redkitchen/cloud_bin_5.pth"

sys.path.insert(0, str(SRC))
os.chdir(SRC)   # needed so relative config paths inside the model resolve

# ─────────────────────────────────────────────────────────────────────────────
# 0. Check no MinkowskiEngine import survives
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/4] Checking MinkowskiEngine is not imported...")
try:
    from models.backbone_kpconv.kpconv import batch_grid_subsampling_kpconv_gpu
    print("      kpconv imported OK (no ME needed)")
except ImportError as e:
    print(f"      FAIL: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit-test the new grid subsampling on GPU
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/4] Unit-testing batch_grid_subsampling_kpconv_gpu...")
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"      Using device: {device}")

# Two fake point clouds of different sizes
torch.manual_seed(0)
pts_a = torch.rand(500, 3, device=device) * 2.0      # 500 pts in [0,2]^3
pts_b = torch.rand(300, 3, device=device) * 2.0      # 300 pts in [0,2]^3
pts   = torch.cat([pts_a, pts_b], dim=0)
lens  = torch.tensor([500, 300], device=device, dtype=torch.int64)

s_pts, s_len = batch_grid_subsampling_kpconv_gpu(pts, lens, sampleDl=0.1)

assert s_pts.device.type == device.type,     "output not on right device"
assert s_len.shape[0] == 2,                  "wrong number of batches returned"
assert s_len.sum() == s_pts.shape[0],        "length mismatch"
assert s_pts.shape[0] < pts.shape[0],        "subsampling should reduce point count"
assert not torch.isnan(s_pts).any(),         "NaN in subsampled points"
print(f"      Input: {pts.shape[0]} pts  →  Output: {s_pts.shape[0]} pts")
print(f"      Per-cloud: {s_len[0].item()} + {s_len[1].item()} = {s_len.sum().item()}")
print("      PASS")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Download pretrained weights if missing
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/4] Checking pretrained weights...")
if not CKPT.exists():
    print("      Downloading trained_models.zip (~200MB)...")
    zip_path = REPO / "trained_models.zip"
    urllib.request.urlretrieve(
        "https://github.com/yewzijian/RegTR/releases/download/v1/trained_models.zip",
        zip_path,
        reporthook=lambda b, bs, total: print(
            f"      {min(b*bs, total)/1e6:.0f}/{total/1e6:.0f} MB", end="\r", flush=True)
    )
    print()
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(REPO)
    print("      Extracted.")
else:
    print(f"      Found: {CKPT}")

assert CKPT.exists(),   f"Checkpoint missing: {CKPT}"
assert SRC_PC.exists(), f"Demo cloud missing: {SRC_PC}"
assert TGT_PC.exists(), f"Demo cloud missing: {TGT_PC}"
print("      PASS")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Full headless forward pass
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/4] Running headless forward pass...")
from pathlib import Path
from easydict import EasyDict
from models.regtr import RegTR
from utils.misc import load_config

cfg   = EasyDict(load_config(Path(CKPT).parents[1] / "config.yaml"))
model = RegTR(cfg).to(device)
# PyTorch 2.6+ defaults weights_only=True; RegTR release pickles need full unpickle.
state = torch.load(str(CKPT), map_location=device, weights_only=False)
model.load_state_dict(state["state_dict"])
model.eval()
print("      Model loaded.")

src_xyz = torch.load(str(SRC_PC), weights_only=False)[:, :3].float().to(device)
tgt_xyz = torch.load(str(TGT_PC), weights_only=False)[:, :3].float().to(device)

with torch.no_grad():
    outputs = model({"src_xyz": [src_xyz], "tgt_xyz": [tgt_xyz]})

pose = outputs["pose"][-1, 0].cpu().numpy()

assert not np.isnan(pose).any(),       "NaN in estimated pose"
assert pose.shape == (4, 4),           f"Wrong pose shape: {pose.shape}"
assert abs(np.linalg.det(pose[:3,:3]) - 1.0) < 0.01, "Rotation matrix det != 1"

print(f"      src: {src_xyz.shape[0]} pts  |  tgt: {tgt_xyz.shape[0]} pts")
print(f"      Estimated pose (4x4):\n{np.round(pose, 4)}")
print("      PASS")

print("\n" + "="*60)
print("  ALL TESTS PASSED — MinkowskiEngine successfully removed")
print("="*60 + "\n")
