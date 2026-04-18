"""Smoke tests for the Task 1 CEFE-1 colour fusion pipeline.

Runs on CPU, no MinkowskiEngine / PyTorch3D / 3DMatch data needed.

What it verifies
----------------
1. ThreeDMatchDataset._split_xyz_rgb correctly decomposes (N,3), (N,6), and
   dict-form payloads, and normalises 0-255 colour to [0,1].
2. _make_rgb honours the color_source policy ('auto', 'random', 'zeros') and
   returns None when use_color is False.
3. collate_pair passes src_rgb / tgt_rgb through as a Python list so the
   downstream RegTR.forward receives them in the same order as src_xyz / tgt_xyz.

Run from src/:
    python tests/test_cefe_color.py
"""
import importlib.util
import os
import sys
import types

import numpy as np
import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ThreeDMatchDataset has heavy top-level imports (h5py, compute_overlap) we'd
# rather skip here. We exercise the two RGB helpers as unbound methods so we
# don't need to instantiate the dataset.
def _load(name, relpath):
    path = os.path.join(_SRC, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub parents so relative imports inside collate_functions resolve.
for pkg in ('data_loaders',):
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [os.path.join(_SRC, pkg)]
        sys.modules[pkg] = m

collate_mod = _load('data_loaders.collate_functions',
                    'data_loaders/collate_functions.py')


class _DummyDS:
    """Minimal stand-in carrying the two RGB helpers off ThreeDMatchDataset."""
    def __init__(self, use_color=True, color_source='auto'):
        self.use_color = use_color
        self.color_source = color_source

    # Copy the methods off the real class without importing it (to dodge h5py).
    from importlib import import_module  # noqa — placeholder


def _bind_helpers():
    """Copy _split_xyz_rgb and _make_rgb off ThreeDMatchDataset, side-stepping
    the module's h5py / compute_overlap imports by faking them."""
    # Fake the heavy deps.
    import importlib, types as _t
    for missing in ('h5py',):
        if missing not in sys.modules:
            sys.modules[missing] = _t.ModuleType(missing)
            sys.modules[missing].File = lambda *a, **k: {}

    # utils.pointcloud.compute_overlap is imported; stub the module if missing.
    utils_pkg = _t.ModuleType('utils')
    utils_pkg.__path__ = [os.path.join(_SRC, 'utils')]
    sys.modules.setdefault('utils', utils_pkg)
    # (actual utils.pointcloud / utils.se3_numpy should import fine on CPU;
    # we only stub the package object to satisfy `from utils.* import ...`.)

    ds_mod = _load('data_loaders.threedmatch', 'data_loaders/threedmatch.py')
    return ds_mod.ThreeDMatchDataset


def test_split_xyz_rgb():
    ThreeDMatchDataset = _bind_helpers()
    ds = _DummyDS()
    ds._split_xyz_rgb = ThreeDMatchDataset._split_xyz_rgb.__get__(ds)

    # (N, 3) → xyz only
    xyz, rgb = ds._split_xyz_rgb(np.random.randn(50, 3).astype(np.float32))
    assert xyz.shape == (50, 3) and rgb is None

    # (N, 6) with colour in [0,255] → normalised
    arr = np.concatenate([np.random.randn(20, 3), np.full((20, 3), 200.0)], axis=1)
    xyz, rgb = ds._split_xyz_rgb(arr)
    assert xyz.shape == (20, 3) and rgb.shape == (20, 3)
    assert 0.0 <= rgb.min() and rgb.max() <= 1.0

    # dict form
    xyz, rgb = ds._split_xyz_rgb({'xyz': np.zeros((5, 3)), 'rgb': np.ones((5, 3)) * 0.5})
    assert rgb.shape == (5, 3) and np.allclose(rgb, 0.5)
    print('[ok] _split_xyz_rgb covers (N,3), (N,6) 0-255, and dict forms')


def test_make_rgb_policies():
    ThreeDMatchDataset = _bind_helpers()

    # use_color off → None regardless of loaded RGB
    ds = _DummyDS(use_color=False)
    ds._make_rgb = ThreeDMatchDataset._make_rgb.__get__(ds)
    assert ds._make_rgb(10, np.zeros((10, 3))) is None

    # color_source='random' → synthetic RGB in [0,1]
    ds = _DummyDS(use_color=True, color_source='random')
    ds._make_rgb = ThreeDMatchDataset._make_rgb.__get__(ds)
    rgb = ds._make_rgb(10, None)
    assert rgb.shape == (10, 3) and 0.0 <= rgb.min() and rgb.max() <= 1.0

    # color_source='zeros' → zeros
    ds = _DummyDS(use_color=True, color_source='zeros')
    ds._make_rgb = ThreeDMatchDataset._make_rgb.__get__(ds)
    rgb = ds._make_rgb(7, np.random.rand(7, 3).astype(np.float32))
    assert rgb.shape == (7, 3) and np.all(rgb == 0)

    # color_source='auto' with loaded RGB → echoes it
    ds = _DummyDS(use_color=True, color_source='auto')
    ds._make_rgb = ThreeDMatchDataset._make_rgb.__get__(ds)
    loaded = np.random.rand(4, 3).astype(np.float32)
    assert np.allclose(ds._make_rgb(4, loaded), loaded)

    # color_source='auto' with no loaded RGB → zeros fallback
    ds._make_rgb = ThreeDMatchDataset._make_rgb.__get__(ds)
    rgb = ds._make_rgb(6, None)
    assert rgb.shape == (6, 3) and np.all(rgb == 0)
    print('[ok] _make_rgb obeys use_color + color_source policy')


def test_collate_passes_rgb():
    collate = collate_mod.collate_pair
    sample = lambda n: {
        'src_xyz': torch.randn(n, 3),
        'tgt_xyz': torch.randn(n, 3),
        'src_rgb': torch.rand(n, 3),
        'tgt_rgb': torch.rand(n, 3),
        'pose': torch.eye(4)[:3],
        'src_overlap': torch.ones(n, dtype=torch.bool),
        'tgt_overlap': torch.ones(n, dtype=torch.bool),
        'correspondences': torch.zeros((0, 2), dtype=torch.int64),
        'src_path': 'a', 'tgt_path': 'b', 'idx': 0, 'overlap_p': 0.5,
    }
    batch = collate([sample(30), sample(45)])
    assert 'src_rgb' in batch and 'tgt_rgb' in batch
    assert isinstance(batch['src_rgb'], list) and len(batch['src_rgb']) == 2
    # Order must match src_xyz (same list-indexing) so downstream stacking lines up.
    assert batch['src_rgb'][0].shape == batch['src_xyz'][0].shape
    print('[ok] collate_pair threads src_rgb / tgt_rgb as a list, aligned with xyz')


if __name__ == '__main__':
    test_split_xyz_rgb()
    test_make_rgb_policies()
    test_collate_passes_rgb()
    print('\nAll CEFE smoke tests passed.')
