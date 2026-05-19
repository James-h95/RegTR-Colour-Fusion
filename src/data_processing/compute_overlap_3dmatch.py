"""Precomputes the overlap regions for 3DMatch dataset,
used for computing the losses in RegTR.
"""
import argparse
import os
import pickle
import shutil
import sys
sys.path.append(os.getcwd())

import h5py
import numpy as np
import torch
from tqdm import tqdm

from utils.pointcloud import compute_overlap
from utils.se3_numpy import se3_transform, se3_init

parser = argparse.ArgumentParser()
# General
parser.add_argument('--base_dir', type=str, default='../data/indoor',
                    help='Path to 3DMatch raw data (Predator format)')
parser.add_argument('--overlap_radius', type=float, default=0.0375,
                    help='Overlap region will be sampled to this voxel size')
opt = parser.parse_args()


def _load_xyz_fragment(path):
    """Load a fragment; return xyz (N, 3) whether stored as (N,3) or (N,6)."""
    raw = torch.load(path)
    arr = raw.detach().cpu().numpy() if torch.is_tensor(raw) else np.asarray(raw)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f'{path}: expected (N,3) or (N,6), got {arr.shape}')
    return arr[:, :3].astype(np.float32)


def process(phase):

    with open(f'datasets/3dmatch/{phase}_info.pkl', 'rb') as fid:
        infos = pickle.load(fid)

    out_file = os.path.join(opt.base_dir, f'{phase}_pairs-overlapmask.h5')
    print(f'Processing {phase}, output: {out_file}...')
    h5_fid = h5py.File(out_file, 'w')

    num_pairs = len(infos['src'])
    for item in tqdm(range(num_pairs)):
        src_path = infos['src'][item]
        tgt_path = infos['tgt'][item]
        pose = se3_init(infos['rot'][item], infos['trans'][item])  # transforms src to tgt

        src_xyz = _load_xyz_fragment(os.path.join(opt.base_dir, src_path))
        tgt_xyz = _load_xyz_fragment(os.path.join(opt.base_dir, tgt_path))

        src_mask, tgt_mask, src_tgt_corr = compute_overlap(
            se3_transform(pose, src_xyz),
            tgt_xyz,
            opt.overlap_radius,
        )

        h5_fid.create_dataset(f'/pair_{item:06d}/src_mask', data=src_mask)
        h5_fid.create_dataset(f'/pair_{item:06d}/tgt_mask', data=tgt_mask)
        h5_fid.create_dataset(f'/pair_{item:06d}/src_tgt_corr', data=src_tgt_corr)


if __name__ == '__main__':
    phases = ['train', 'val', 'test_3DMatch', 'test_3DLoMatch']
    # Color benchmarks use the same pair lists as 3DMatch / 3DLoMatch.
    for color_phase, base_phase in [
        ('test_Color3DMatch', 'test_3DMatch'),
        ('test_Color3DLoMatch', 'test_3DLoMatch'),
    ]:
        if os.path.exists(f'datasets/3dmatch/{color_phase}_info.pkl'):
            phases.append(color_phase)
        elif os.path.exists(f'datasets/3dmatch/{base_phase}_info.pkl'):
            shutil.copy(
                f'datasets/3dmatch/{base_phase}_info.pkl',
                f'datasets/3dmatch/{color_phase}_info.pkl',
            )
            phases.append(color_phase)
    for phase in phases:
        process(phase)