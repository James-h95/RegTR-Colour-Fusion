"""Dataloader for 3DMatch dataset

Modified from Predator source code by Shengyu Huang:
  https://github.com/overlappredator/OverlapPredator/blob/main/datasets/indoor.py
"""
import logging
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.se3_numpy import se3_init, se3_transform, se3_inv
from utils.pointcloud import compute_overlap


class ThreeDMatchDataset(Dataset):

    def __init__(self, cfg, phase, transforms=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        assert phase in ['train', 'val', 'test']
        if phase in ['train', 'val']:
            info_fname = f'datasets/3dmatch/{phase}_info.pkl'
            pairs_fname = f'{phase}_pairs-overlapmask.h5'
        else:
            info_fname = f'datasets/3dmatch/{phase}_{cfg.benchmark}_info.pkl'
            pairs_fname = f'{phase}_{cfg.benchmark}_pairs-overlapmask.h5'

        with open(info_fname, 'rb') as fid:
            self.infos = pickle.load(fid)

        self.base_dir = None
        if isinstance(cfg.root, str):
            if os.path.exists(f'{cfg.root}/train'):
                self.base_dir = cfg.root
        else:
            for r in cfg.root:
                if os.path.exists(f'{r}/train'):
                    self.base_dir = r
                break
        if self.base_dir is None:
            raise AssertionError(f'Dataset not found in {cfg.root}')
        else:
            self.logger.info(f'Loading data from {self.base_dir}')

        self.cfg = cfg

        # Optional fallback root: if a fragment is missing from base_dir (e.g.
        # Color3DMatch doesn't include every 3DMatch scene) we fall back to this
        # directory for xyz-only data. Set cfg.fallback_root to '../data/indoor'.
        fallback = cfg.get('fallback_root', None)
        self.fallback_dir = fallback if (fallback and os.path.exists(f'{fallback}/train')) else None
        if self.fallback_dir:
            self.logger.info(f'Fallback root: {self.fallback_dir}')

        if os.path.exists(os.path.join(self.base_dir, pairs_fname)):
            self.pairs_data = h5py.File(os.path.join(self.base_dir, pairs_fname), 'r')
        else:
            self.logger.warning(
                'Overlapping regions not precomputed. '
                'Run data_processing/compute_overlap_3dmatch.py to speed up data loading')
            self.pairs_data = None

        self.search_voxel_size = cfg.overlap_radius
        self.transforms = transforms
        self.phase = phase

        # Colour fusion (Task 1, CEFE). When cfg.use_color is True we try to
        # obtain a per-point RGB tensor; the model consumes it as the initial
        # KPConv features instead of the geometry-blind ones.
        self.use_color = bool(cfg.get('use_color', False))
        # 'auto'    : split from the xyz file if it's (N,6), else zeros
        # 'random'  : synthesise random RGB (for architecture sanity / pretraining
        #             the colour pathway when Color3DMatch isn't available yet)
        # 'zeros'   : always zeros (equivalent to ones but matches RGB dim)
        self.color_source = cfg.get('color_source', 'auto')

    def __len__(self):
        return len(self.infos['rot'])

    def _split_xyz_rgb(self, raw, n_points_hint=None):
        """Given a raw loaded point payload return (xyz[N,3], rgb[N,3] or None).

        Supports:
          * numpy/tensor of shape (N, 3)               -> xyz only
          * numpy/tensor of shape (N, 6)               -> xyz + rgb (rgb scaled
                                                         to [0,1] if it looks
                                                         like 0-255)
          * dict with keys {'xyz', 'rgb'}              -> explicit colour
        """
        if isinstance(raw, dict):
            xyz = np.asarray(raw['xyz'])
            rgb = np.asarray(raw['rgb']) if 'rgb' in raw else None
        else:
            arr = np.asarray(raw)
            if arr.ndim == 2 and arr.shape[1] >= 6:
                xyz, rgb = arr[:, :3], arr[:, 3:6]
            else:
                xyz, rgb = arr, None
        if rgb is not None:
            rgb = rgb.astype(np.float32)
            if rgb.max() > 1.5:  # heuristic: stored as 0-255
                rgb = rgb / 255.0
            rgb = np.clip(rgb, 0.0, 1.0)
        return xyz, rgb

    def _make_rgb(self, N, loaded_rgb):
        """Resolve the RGB tensor per the configured color_source policy."""
        if not self.use_color:
            return None
        if self.color_source == 'random':
            return np.random.rand(N, 3).astype(np.float32)
        if self.color_source == 'zeros' or loaded_rgb is None:
            return np.zeros((N, 3), dtype=np.float32)
        return loaded_rgb.astype(np.float32)

    def __getitem__(self, item):

        # get transformation and point cloud
        pose = se3_init(self.infos['rot'][item], self.infos['trans'][item])  # transforms src to tgt
        pose_inv = se3_inv(pose)
        src_path = self.infos['src'][item]
        tgt_path = self.infos['tgt'][item]

        def _load_fragment(rel_path):
            full = os.path.join(self.base_dir, rel_path)
            if not os.path.exists(full) and self.fallback_dir:
                full = os.path.join(self.fallback_dir, rel_path)
            return torch.load(full, weights_only=False)

        src_raw = _load_fragment(src_path)
        tgt_raw = _load_fragment(tgt_path)
        src_xyz, src_rgb_loaded = self._split_xyz_rgb(src_raw)
        tgt_xyz, tgt_rgb_loaded = self._split_xyz_rgb(tgt_raw)
        src_rgb = self._make_rgb(src_xyz.shape[0], src_rgb_loaded)
        tgt_rgb = self._make_rgb(tgt_xyz.shape[0], tgt_rgb_loaded)
        overlap_p = self.infos['overlap'][item]

        # Get overlap region
        if self.pairs_data is None:
            src_overlap_mask, tgt_overlap_mask, src_tgt_corr = compute_overlap(
                se3_transform(pose, src_xyz),
                tgt_xyz,
                self.search_voxel_size,
            )
        else:
            src_overlap_mask = np.asarray(self.pairs_data[f'pair_{item:06d}/src_mask'])
            tgt_overlap_mask = np.asarray(self.pairs_data[f'pair_{item:06d}/tgt_mask'])
            src_tgt_corr = np.asarray(self.pairs_data[f'pair_{item:06d}/src_tgt_corr'])

        data_pair = {
            'src_xyz': torch.from_numpy(src_xyz).float(),
            'tgt_xyz': torch.from_numpy(tgt_xyz).float(),
            'src_overlap': torch.from_numpy(src_overlap_mask),
            'tgt_overlap': torch.from_numpy(tgt_overlap_mask),
            'correspondences': torch.from_numpy(src_tgt_corr),  # indices
            'pose': torch.from_numpy(pose).float(),
            'idx': item,
            'src_path': src_path,
            'tgt_path': tgt_path,
            'overlap_p': overlap_p,
        }
        if src_rgb is not None:
            data_pair['src_rgb'] = torch.from_numpy(src_rgb).float()
            data_pair['tgt_rgb'] = torch.from_numpy(tgt_rgb).float()

        if self.transforms is not None:
            self.transforms(data_pair)  # Apply data augmentation

        return data_pair
