"""Smoke tests for Task 2 relational positional encoding (Q·R Tier 3).

Runs on CPU, no MinkowskiEngine / PyTorch3D / 3DMatch data needed.

What it verifies
----------------
1. GeometricStructureEmbedding forward produces (N, N, head_dim) per sample.
2. Output is approximately invariant to rigid transforms of input points.
3. _pack_sa_relation pads to (B, N_max, N_max, head_dim).
4. TransformerCrossEncoderLayer runs end-to-end with relation threaded through
   self-attention (and still runs without it).

Run from src/:
    python -m tests.test_relational_pos_emb
or:
    python tests/test_relational_pos_emb.py
"""
import importlib.util
import os
import sys
import types

import torch

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _load(name: str, relpath: str):
    path = os.path.join(_SRC, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for pkg in ('models', 'models.transformer'):
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [os.path.join(_SRC, *pkg.split('.'))]
        sys.modules[pkg] = m

pe_mod = _load('models.transformer.position_embedding',
               'models/transformer/position_embedding.py')
tf_mod = _load('models.transformer.transformers',
               'models/transformer/transformers.py')

GeometricStructureEmbedding = pe_mod.GeometricStructureEmbedding
TransformerCrossEncoderLayer = tf_mod.TransformerCrossEncoderLayer


def _random_rigid(device):
    A = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    Q = Q @ torch.diag(torch.sign(torch.diagonal(R)))
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = torch.randn(3, device=device) * 2.0
    return Q, t


def _pack_sa_relation(rel_list):
    """Mirror of models.regtr._pack_sa_relation."""
    if len(rel_list) == 0:
        return None
    head_dim = rel_list[0].shape[-1]
    N_max = max(r.shape[0] for r in rel_list)
    device, dtype = rel_list[0].device, rel_list[0].dtype
    padded = torch.zeros(len(rel_list), N_max, N_max, head_dim,
                         device=device, dtype=dtype)
    for i, r in enumerate(rel_list):
        n = r.shape[0]
        padded[i, :n, :n, :] = r
    return padded


def test_shape():
    torch.manual_seed(0)
    d_model, nhead = 256, 8
    head_dim = d_model // nhead
    m = GeometricStructureEmbedding(
        hidden_dim=64, num_heads=nhead, d_model=d_model)
    pts = [torch.randn(120, 3), torch.randn(90, 3)]
    out = m(pts)
    assert len(out) == 2
    assert out[0].shape == (120, 120, head_dim), f'got {out[0].shape}'
    assert out[1].shape == (90, 90, head_dim), f'got {out[1].shape}'
    assert torch.isfinite(out[0]).all(), 'NaN/Inf in relation tensor'
    print('[ok] shape & finiteness')


def test_rigid_invariance():
    torch.manual_seed(1)
    d_model, nhead = 128, 4
    head_dim = d_model // nhead
    m = GeometricStructureEmbedding(
        hidden_dim=64, num_heads=nhead, d_model=d_model)
    m.eval()
    pts = torch.randn(80, 3)

    Rm, t = _random_rigid(pts.device)
    pts_trans = pts @ Rm.T + t

    with torch.no_grad():
        r1 = m([pts])[0]
        r2 = m([pts_trans])[0]

    diff = (r1 - r2).abs().max().item()
    assert diff < 1e-3, f'relation not rigid-invariant (max diff={diff:.2e})'
    print(f'[ok] rigid invariance (max diff={diff:.2e})')


def test_pack_sa_relation():
    head_dim = 32
    rels = [torch.randn(100, 100, head_dim), torch.randn(60, 60, head_dim)]
    packed = _pack_sa_relation(rels)
    assert packed.shape == (2, 100, 100, head_dim), f'got {packed.shape}'
    assert torch.all(packed[1, 60:, :, :] == 0), 'sample-1 row padding'
    assert torch.all(packed[1, :, 60:, :] == 0), 'sample-1 col padding'
    print('[ok] _pack_sa_relation padding & shape')


def test_transformer_layer_with_relation():
    torch.manual_seed(2)
    d_model, nhead = 64, 4
    head_dim = d_model // nhead
    layer = TransformerCrossEncoderLayer(
        d_model, nhead, dim_feedforward=128,
        dropout=0.0, normalize_before=True)
    layer.eval()

    N_pad, B = 50, 2
    src = torch.randn(N_pad, B, d_model)
    tgt = torch.randn(N_pad, B, d_model)

    geo = GeometricStructureEmbedding(
        hidden_dim=32, num_heads=nhead, d_model=d_model)
    src_pts = [torch.randn(N_pad, 3) for _ in range(B)]
    tgt_pts = [torch.randn(N_pad, 3) for _ in range(B)]
    src_rel = _pack_sa_relation(geo(src_pts))
    tgt_rel = _pack_sa_relation(geo(tgt_pts))

    out_s0, out_t0 = layer(src, tgt)
    out_s1, out_t1 = layer(
        src, tgt,
        src_sa_relation=src_rel,
        tgt_sa_relation=tgt_rel)

    assert out_s0.shape == out_s1.shape == src.shape
    assert out_t0.shape == out_t1.shape == tgt.shape
    assert not torch.allclose(out_s0, out_s1, atol=1e-6), \
        'relation had no effect on src self-attn output'
    print('[ok] transformer layer runs with & without relation; Q·R changes output')


if __name__ == '__main__':
    test_shape()
    test_rigid_invariance()
    test_pack_sa_relation()
    test_transformer_layer_with_relation()
    print('\nAll smoke tests passed.')
