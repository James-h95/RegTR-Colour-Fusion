"""Smoke test for the Color3DMatch conversion script.

Creates a temporary directory of fake .npy fragments (both (N,3) and (N,6)
variants, with both 0-1 and 0-255 colour ranges), runs the conversion,
and verifies the resulting .pth files round-trip through
ThreeDMatchDataset._split_xyz_rgb.

Run from src/:
    python tests/test_color3dmatch_conversion.py
"""
import importlib.util
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _load(name, relpath):
    path = os.path.join(_SRC, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub parent packages so the conversion script's relative helper imports
# resolve without dragging in h5py / etc.
for pkg in ('data_processing', 'data_loaders'):
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [os.path.join(_SRC, pkg)]
        sys.modules[pkg] = m

conv_mod = _load('data_processing.convert_color3dmatch',
                 'data_processing/convert_color3dmatch.py')

# Fake h5py so threedmatch.py imports.
sys.modules.setdefault('h5py', types.ModuleType('h5py'))
sys.modules['h5py'].File = lambda *a, **k: {}
utils_pkg = types.ModuleType('utils')
utils_pkg.__path__ = [os.path.join(_SRC, 'utils')]
sys.modules.setdefault('utils', utils_pkg)

ds_mod = _load('data_loaders.threedmatch', 'data_loaders/threedmatch.py')
ThreeDMatchDataset = ds_mod.ThreeDMatchDataset


def _make_tree(src_root: Path):
    scenes = {
        'train/7-scenes-chess/cloud_bin_0.npy': (500, 6, '0-1'),
        'train/7-scenes-chess/cloud_bin_1.npy': (300, 6, '0-255'),
        'train/sun3d-hotel/cloud_bin_0.npy':    (450, 3, 'n/a'),
        'test/7-scenes-fire/cloud_bin_2.npy':   (600, 6, '0-1'),
    }
    meta = {}
    for rel, (n, cols, scale) in scenes.items():
        path = src_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = np.random.randn(n, 3).astype(np.float32)
        if cols == 6:
            if scale == '0-1':
                rgb = np.random.rand(n, 3).astype(np.float32)
            else:
                rgb = (np.random.rand(n, 3) * 255).astype(np.float32)
            arr = np.concatenate([xyz, rgb], axis=1)
        else:
            arr = xyz
        np.save(str(path), arr)
        meta[rel] = (cols, scale)
    return meta


def test_conversion_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / 'color3dmatch'
        dst = Path(tmp) / 'indoor'
        meta = _make_tree(src)

        conv_mod.walk_and_convert(src, dst, dry_run=False, verify=False)

        # All .npy should have a .pth twin at the matching relative path.
        for rel in meta:
            pth = (dst / rel).with_suffix('.pth')
            assert pth.exists(), f'missing converted file: {pth}'

        # Round-trip each file through _split_xyz_rgb.
        dummy = type('D', (), {'_split_xyz_rgb': ThreeDMatchDataset._split_xyz_rgb})()
        for rel, (cols, scale) in meta.items():
            pth = (dst / rel).with_suffix('.pth')
            raw = torch.load(str(pth))
            xyz, rgb = dummy._split_xyz_rgb(raw)
            assert xyz.shape[1] == 3
            if cols == 6:
                assert rgb is not None, f'{rel}: expected rgb, got None'
                assert rgb.shape == xyz.shape
                assert 0.0 <= rgb.min() and rgb.max() <= 1.0, \
                    f'{rel}: rgb out of [0,1] (min={rgb.min()}, max={rgb.max()}) ' \
                    f'(source was {scale})'
            else:
                assert rgb is None
        print(f'[ok] converted {len(meta)} fake fragments, round-trip passes')


def test_idempotent_skip():
    """A second conversion run should skip already-converted files."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / 'color3dmatch'
        dst = Path(tmp) / 'indoor'
        _make_tree(src)
        conv_mod.walk_and_convert(src, dst, dry_run=False, verify=False)
        mtime_first = {p: p.stat().st_mtime for p in dst.rglob('*.pth')}
        conv_mod.walk_and_convert(src, dst, dry_run=False, verify=False)
        for p, mt in mtime_first.items():
            assert p.stat().st_mtime == mt, f'{p} was rewritten — skip logic broken'
        print('[ok] re-running conversion skips existing .pth files')


if __name__ == '__main__':
    test_conversion_roundtrip()
    test_idempotent_skip()
    print('\nAll conversion smoke tests passed.')
