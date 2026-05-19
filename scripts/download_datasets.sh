#!/usr/bin/env bash
# Download 3DMatch (Predator) + Color3DMatch (ColorPCR) for RegTR-Colour-Fusion.
#
# Run from repo root on the pod (needs ~15–25 GB disk, good network):
#   conda activate regtr
#   bash scripts/download_datasets.sh
#
# Optional env vars:
#   REPO_ROOT      — default: parent of scripts/
#   SKIP_PREDATOR  — set 1 to skip 3DMatch download
#   SKIP_COLOR     — set 1 to skip Color3DMatch download
#   SKIP_OVERLAP   — set 1 to skip overlap precompute (slow, ~hours on CPU)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DL="$REPO_ROOT/downloads"
DATA="$REPO_ROOT/data"
mkdir -p "$DL" "$DATA/indoor" "$DATA/indoor_color"

PREDATOR_URL="${PREDATOR_URL:-https://share.phys.ethz.ch/~gseg/Predator/data.zip}"
COLOR_GDRIVE_ID="${COLOR_GDRIVE_ID:-1pQEo0086ipWwNrroAk_ybnhKildq4o_j}"

log() { echo "[download] $*"; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Missing command: $1"; exit 1; }
}

# ── 1) Predator / 3DMatch (xyz .pth) ─────────────────────────────────────────
if [[ "${SKIP_PREDATOR:-0}" != "1" ]]; then
  need_cmd wget
  need_cmd unzip
  if [[ ! -d "$DATA/indoor/train" ]] || [[ -z "$(ls -A "$DATA/indoor/train" 2>/dev/null || true)" ]]; then
    log "Downloading Predator 3DMatch (~5–6 GB) ..."
    wget -c --no-check-certificate -O "$DL/predator_data.zip" "$PREDATOR_URL"
    log "Unzipping Predator data ..."
    unzip -qo "$DL/predator_data.zip" -d "$DL/predator_unzip"
    # data.zip unpacks as data/indoor/{train,val,test}
    if [[ -d "$DL/predator_unzip/data/indoor" ]]; then
      rsync -a "$DL/predator_unzip/data/indoor/" "$DATA/indoor/"
    elif [[ -d "$DL/predator_unzip/indoor" ]]; then
      rsync -a "$DL/predator_unzip/indoor/" "$DATA/indoor/"
    else
      echo "Unexpected Predator zip layout under $DL/predator_unzip"
      find "$DL/predator_unzip" -maxdepth 3 -type d | head -20
      exit 1
    fi
    log "3DMatch fragments -> $DATA/indoor"
  else
    log "Skip Predator download (indoor/train already present)"
  fi
else
  log "SKIP_PREDATOR=1"
fi

# ── 2) Color3DMatch (ColorPCR Google Drive) ───────────────────────────────────
if [[ "${SKIP_COLOR:-0}" != "1" ]]; then
  if [[ ! -d "$DATA/indoor_color/train" ]] || [[ -z "$(ls -A "$DATA/indoor_color/train" 2>/dev/null || true)" ]]; then
    python -m pip install -q gdown
    log "Downloading Color3DMatch from ColorPCR Google Drive (large, ~10+ GB) ..."
    gdown --id "$COLOR_GDRIVE_ID" -O "$DL/color3dmatch.zip"
    log "Unzipping Color3DMatch ..."
    unzip -qo "$DL/color3dmatch.zip" -d "$DL/color_unzip"
    # ColorPCR layout: dataset/data/{train,test}/<scene>/cloud_bin_*.npy
    COLOR_SRC=""
    for candidate in \
      "$DL/color_unzip/dataset/data" \
      "$DL/color_unzip/data" \
      "$DL/color_unzip"; do
      if [[ -d "$candidate/train" ]]; then
        COLOR_SRC="$candidate"
        break
      fi
    done
    if [[ -z "$COLOR_SRC" ]]; then
      echo "Could not find train/ under Color3DMatch zip. Inspect:"
      find "$DL/color_unzip" -maxdepth 4 -type d | head -30
      exit 1
    fi
    log "Converting .npy -> .pth under $DATA/indoor_color ..."
    (
      cd "$REPO_ROOT/src"
      python -m data_processing.convert_color3dmatch \
        --src "$COLOR_SRC" \
        --dst "$DATA/indoor_color" \
        --verify
    )
  else
    log "Skip Color3DMatch download (indoor_color/train already present)"
  fi
else
  log "SKIP_COLOR=1"
fi

# ── 3) Color benchmark metadata (same pairs as 3DMatch / 3DLoMatch) ─────────────
PKL_DIR="$REPO_ROOT/src/datasets/3dmatch"
for pair in \
  "test_3DMatch_info.pkl:test_Color3DMatch_info.pkl" \
  "test_3DLoMatch_info.pkl:test_Color3DLoMatch_info.pkl"; do
  src="${pair%%:*}"
  dst="${pair##*:}"
  if [[ -f "$PKL_DIR/$src" ]] && [[ ! -f "$PKL_DIR/$dst" ]]; then
    cp "$PKL_DIR/$src" "$PKL_DIR/$dst"
    log "Created $dst (copy of $src)"
  fi
done

# ── 4) Overlap masks (train / val / test) ─────────────────────────────────────
if [[ "${SKIP_OVERLAP:-0}" != "1" ]]; then
  need_cmd python
  log "Precomputing overlap masks for data/indoor (xyz) ..."
  (
    cd "$REPO_ROOT/src"
    python data_processing/compute_overlap_3dmatch.py --base_dir "../data/indoor"
  )
  # Geometry matches Color3DMatch xyz; reuse masks (saves hours).
  for f in train_pairs-overlapmask.h5 val_pairs-overlapmask.h5 \
           test_3DMatch_pairs-overlapmask.h5 test_3DLoMatch_pairs-overlapmask.h5; do
    if [[ -f "$DATA/indoor/$f" ]] && [[ ! -e "$DATA/indoor_color/$f" ]]; then
      ln -sf "../indoor/$f" "$DATA/indoor_color/$f"
      log "Linked indoor_color/$f -> indoor/$f"
    fi
  done
  for f in test_Color3DMatch_pairs-overlapmask.h5 test_Color3DLoMatch_pairs-overlapmask.h5; do
    src="${f/Color3D/3D}"
    if [[ -f "$DATA/indoor/$src" ]] && [[ ! -e "$DATA/indoor_color/$f" ]]; then
      ln -sf "../indoor/$src" "$DATA/indoor_color/$f"
      log "Linked indoor_color/$f -> indoor/$src"
    fi
  done
else
  log "SKIP_OVERLAP=1 (training will work but slower without .h5 masks)"
fi

log "Done. Expected layout:"
log "  $DATA/indoor/{train,val,test}/.../*.pth"
log "  $DATA/indoor/*_pairs-overlapmask.h5"
log "  $DATA/indoor_color/{train,test}/.../*.pth  (N,6) xyz+rgb"
log "  src/datasets/3dmatch/*_info.pkl (+ test_Color*_info.pkl)"
