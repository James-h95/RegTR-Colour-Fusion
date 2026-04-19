"""Convert ColorPCR's Color3DMatch / Color3DLoMatch .npy fragments to the
.pth format RegTR's dataloader consumes.

ColorPCR ships (N, 6) float32 arrays (XYZ + RGB) as .npy files. RegTR's
ThreeDMatchDataset calls torch.load() on each fragment path listed in its
*_info.pkl files. This script walks a source tree and rewrites each .npy
as a .pth at the matching relative path, preserving the (N, 6) layout so
our _split_xyz_rgb helper picks up both XYZ and RGB.

Typical usage
-------------
    # Convert the whole tree in-place alongside the .npy files
    python -m data_processing.convert_color3dmatch \\
        --src /path/to/color3dmatch \\
        --dst ../data/indoor \\
        --verify

    # Dry run (lists what would happen, no writes)
    python -m data_processing.convert_color3dmatch \\
        --src /path/to/color3dmatch --dst ../data/indoor --dry_run

Assumptions
-----------
* Color3DMatch preserves the scene-name layout used by 3DMatch/Predator
  (e.g. 'train/7-scenes-chess/cloud_bin_0.npy'), so REGTR's existing
  train_info.pkl / val_info.pkl / test_<benchmark>_info.pkl will resolve
  after just a rename. If ColorPCR changed the scene split, you'll need to
  rebuild the info pkls separately — see README in this directory.
* Colour is stored linearly in [0, 1] or [0, 255]. Values >1.5 are treated
  as 0-255 and normalised.

After running this, set the following in your config:
    dataset:
        root: '../data/indoor'
        use_color: True
        color_source: auto   # <-- splits (N,6) back into xyz + rgb
    kpconv_options:
        in_feats_dim: 3
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

_logger = logging.getLogger(__name__)


def convert_one(npy_path: Path, pth_path: Path, dry_run: bool = False) -> dict:
    """Load a single .npy fragment and save it as .pth.

    Returns a small stats dict for summary reporting.
    """
    arr = np.load(str(npy_path))
    if arr.ndim != 2 or arr.shape[1] not in (3, 6):
        raise ValueError(
            f'{npy_path}: unexpected shape {arr.shape} '
            f'(expected (N,3) or (N,6))')
    has_rgb = arr.shape[1] == 6
    if has_rgb:
        rgb = arr[:, 3:6]
        if rgb.max() > 1.5:
            arr = arr.copy()
            arr[:, 3:6] = np.clip(rgb / 255.0, 0.0, 1.0)
    arr = arr.astype(np.float32)
    if not dry_run:
        pth_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as torch.Tensor (rather than raw numpy) so PyTorch >=2.6 can
        # load with the safer weights_only=True default. Our _split_xyz_rgb
        # calls np.asarray() on whatever comes back, so a tensor round-trips
        # transparently; the legacy Predator-format pkls REGTR ships with
        # were saved as numpy arrays, which is what required weights_only=False.
        torch.save(torch.from_numpy(arr), str(pth_path))
    return {'n_points': int(arr.shape[0]), 'has_rgb': has_rgb}


def walk_and_convert(src: Path, dst: Path, dry_run: bool, verify: bool):
    n_converted = 0
    n_skipped = 0
    n_rgb = 0
    total_points = 0
    for npy_path in src.rglob('*.npy'):
        rel = npy_path.relative_to(src)
        pth_path = (dst / rel).with_suffix('.pth')
        if pth_path.exists():
            _logger.debug(f'skip (exists): {pth_path}')
            n_skipped += 1
            continue
        try:
            stats = convert_one(npy_path, pth_path, dry_run=dry_run)
        except Exception as e:
            _logger.error(f'FAIL {npy_path}: {e}')
            continue
        n_converted += 1
        total_points += stats['n_points']
        n_rgb += int(stats['has_rgb'])
        if n_converted % 200 == 0:
            _logger.info(f'converted {n_converted} files...')
    _logger.info(
        f'Done. converted={n_converted}, skipped={n_skipped}, '
        f'with_rgb={n_rgb}, avg_points={total_points // max(n_converted,1)}')

    if verify and not dry_run and n_converted > 0:
        _verify_roundtrip(dst)


def _verify_roundtrip(dst: Path):
    """Load 3 random converted files, split via _split_xyz_rgb, sanity-check."""
    try:
        # Import the helper off ThreeDMatchDataset WITHOUT triggering the
        # heavy package imports (h5py etc.) by using an unbound reference.
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from data_loaders.threedmatch import ThreeDMatchDataset
    except Exception as e:
        _logger.warning(f'Could not import ThreeDMatchDataset for verify: {e}')
        return
    samples = list(dst.rglob('*.pth'))[:3]
    for p in samples:
        raw = torch.load(str(p))
        xyz, rgb = ThreeDMatchDataset._split_xyz_rgb(None, raw)
        assert xyz.shape[1] == 3, f'{p}: bad xyz shape {xyz.shape}'
        rgb_info = 'no-rgb' if rgb is None else f'rgb[{rgb.min():.2f},{rgb.max():.2f}]'
        _logger.info(f'verify {p.name}: xyz={xyz.shape} {rgb_info}')
    _logger.info('verify OK')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True, type=Path,
                        help='Root of the Color3DMatch .npy tree')
    parser.add_argument('--dst', required=True, type=Path,
                        help='Output root (typically ../data/indoor)')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--verify', action='store_true',
                        help='After conversion, round-trip-load a few files')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s')

    if not args.src.exists():
        _logger.error(f'src does not exist: {args.src}')
        sys.exit(1)
    walk_and_convert(args.src, args.dst, args.dry_run, args.verify)


if __name__ == '__main__':
    main()
