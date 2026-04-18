"""Smoke tests for the Task 2 relational positional encoding.

Runs on CPU, no MinkowskiEngine / PyTorch3D / 3DMatch data needed.

What it verifies
----------------
1. GeometricStructureEmbedding forward produces (num_heads, N, N) bias per sample.
2. The bias is approximately invariant to rigid transforms (rotation + translation)
   — this is the whole point of relational encoding.
3. _pack_sa_bias produces the (B*num_heads, N_max, N_max) shape
   nn.MultiheadAttention expects.
4. TransformerCrossEncoderLayer runs end-to-end with the bias threaded through
   self-attention (and still runs without it — baseline is unbroken).

Run from src/:
    python -m tests.test_relational_pos_emb
or:
    python tests/test_relational_pos_emb.py
"""
import importlib.util
import math
import os
import sys
import types

import torch

# Make `src/` importable when run as a bare script.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _load(name: str, relpath: str):
    """Import a file as `name` without triggering package __init__.py chains.

    Needed because `models/__init__.py` auto-imports regtr.py, which pulls in
    pytorch3d + MinkowskiEngine. We only want the transformer submodule here.
    """
    path = os.path.join(_SRC, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub parent packages so relative imports inside the target files resolve.
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
    # Random rotation via QR.
    A = torch.randn(3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    # Ensure det(Q) = +1 (proper rotation).
    Q = Q @ torch.diag(torch.sign(torch.diagonal(R)))
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    t = torch.randn(3, device=device) * 2.0
    return Q, t


def test_shape():
    torch.manual_seed(0)
    m = GeometricStructureEmbedding(hidden_dim=64, num_heads=8)
    pts = [torch.randn(120, 3), torch.randn(90, 3)]
    out = m(pts)
    assert len(out) == 2
    assert out[0].shape == (8, 120, 120), f'got {out[0].shape}'
    assert out[1].shape == (8, 90, 90), f'got {out[1].shape}'
    assert torch.isfinite(out[0]).all(), 'NaN/Inf in bias'
    print('[ok] shape & finiteness')


def test_rigid_invariance():
    """Distances and triplet angles are rigid-invariant, so the bias should be too."""
    torch.manual_seed(1)
    m = GeometricStructureEmbedding(hidden_dim=64, num_heads=4)
    m.eval()
    pts = torch.randn(80, 3)

    R, t = _random_rigid(pts.device)
    pts_trans = pts @ R.T + t

    with torch.no_grad():
        b1 = m([pts])[0]
        b2 = m([pts_trans])[0]

    diff = (b1 - b2).abs().max().item()
    # Not exactly zero due to float error + top-k ties; should be tiny.
    assert diff < 1e-3, f'bias not rigid-invariant (max diff={diff:.2e})'
    print(f'[ok] rigid invariance (max diff={diff:.2e})')


def _pack_sa_bias(bias_list):
    # Mirror of models.regtr._pack_sa_bias — re-declared here to avoid
    # importing regtr.py (which would drag in pytorch3d + Minkowski).
    if len(bias_list) == 0:
        return None
    H = bias_list[0].shape[0]
    N_max = max(b.shape[-1] for b in bias_list)
    device, dtype = bias_list[0].device, bias_list[0].dtype
    padded = torch.zeros(len(bias_list), H, N_max, N_max, device=device, dtype=dtype)
    for i, b in enumerate(bias_list):
        n = b.shape[-1]
        padded[i, :, :n, :n] = b
    return padded.reshape(len(bias_list) * H, N_max, N_max)


def test_pack_sa_bias():
    biases = [torch.randn(8, 100, 100), torch.randn(8, 60, 60)]
    packed = _pack_sa_bias(biases)
    assert packed.shape == (16, 100, 100), f'got {packed.shape}'
    # Sample 0 should be padded in bottom-right block.
    assert (packed[:8, 60:, :] != 0).any(), 'sample-0 content should live in [:100, :100]'
    # Sample 1 padding region should be zeros.
    assert torch.all(packed[8:, 60:, :] == 0), 'sample-1 padding should be zero'
    assert torch.all(packed[8:, :, 60:] == 0), 'sample-1 padding should be zero'
    print('[ok] _pack_sa_bias padding & shape')


def test_transformer_layer_with_bias():
    """Full self-attn + cross-attn layer runs with the geo bias threaded in."""
    torch.manual_seed(2)
    d_model, nhead = 64, 4
    layer = TransformerCrossEncoderLayer(d_model, nhead, dim_feedforward=128,
                                         dropout=0.0, normalize_before=True)
    layer.eval()

    N_src, N_tgt, B = 50, 40, 2
    src = torch.randn(max(N_src, N_tgt), B, d_model)  # padded to max
    tgt = torch.randn(max(N_src, N_tgt), B, d_model)

    # Build per-sample biases (pretend both samples are full length for simplicity).
    geo = GeometricStructureEmbedding(hidden_dim=32, num_heads=nhead)
    src_pts = [torch.randn(max(N_src, N_tgt), 3) for _ in range(B)]
    tgt_pts = [torch.randn(max(N_src, N_tgt), 3) for _ in range(B)]
    src_bias = _pack_sa_bias(geo(src_pts))
    tgt_bias = _pack_sa_bias(geo(tgt_pts))

    # Baseline path: no bias.
    out_s0, out_t0 = layer(src, tgt)
    # Relational path: with bias.
    out_s1, out_t1 = layer(src, tgt,
                           src_sa_attn_bias=src_bias,
                           tgt_sa_attn_bias=tgt_bias)

    assert out_s0.shape == out_s1.shape == src.shape
    assert out_t0.shape == out_t1.shape == tgt.shape
    # The two paths should produce *different* outputs (bias has effect).
    assert not torch.allclose(out_s0, out_s1, atol=1e-6), \
        'bias had no effect on src self-attn output'
    print('[ok] transformer layer runs with & without bias; bias changes output')


if __name__ == '__main__':
    test_shape()
    test_rigid_invariance()
    test_pack_sa_bias()
    test_transformer_layer_with_bias()
    print('\nAll smoke tests passed.')
