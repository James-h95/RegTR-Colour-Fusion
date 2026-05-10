import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalScalarEmbedding(nn.Module):
    """Sinusoidal embedding for a scalar (e.g. pairwise distance / angle)."""

    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0, 'd_model must be even'
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(temperature) / d_model)
        )
        self.register_buffer('div_term', div_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1) * self.div_term  # (..., d_model/2)
        emb = torch.stack([x.sin(), x.cos()], dim=-1).flatten(-2)
        return emb


class GeometricStructureEmbedding(nn.Module):
    """GeoTransformer-style relational positional encoding (Tier 3, Q·R port).

    Produces a per-pair geometric embedding suitable for the FULL GeoTransformer
    geometric self-attention formulation:

        attn[i,j] = (Q_i · K_j  +  Q_i · R_ij) / sqrt(d)

    where R_ij is a per-pair embedding projected into key (head_dim) space.
    This is materially different from the simplified additive-logit-bias port
    we used previously (`attn[i,j] = Q_i · K_j + bias[i,j]`): in this Q·R
    formulation the relational contribution depends on the query content,
    so different queries can attend to different geometric patterns. The
    additive bias couldn't express that — every query saw the same scalar
    bias for a given (i,j) pair, which is why the simplified port was
    expressively crippled and contributed ~0pp on top of the colour
    pathway in our 5-epoch ablation.

    R is built from rigid-invariant primitives:
      * Pairwise distance embeddings (PDE).
      * Triplet-angle embeddings (TAE) aggregated over each anchor's k-NN.

    Shared across heads (matches GeoTransformer's published code: a single
    head-shared geometric prior keeps memory bounded and lets per-head
    diversity come from the per-head Q projections).

    forward(points_list) returns a list of tensors shaped (N_i, N_i, head_dim)
    — one per sample. RegTR's _pack_sa_relation pads the list into a single
    (B, N_max, N_max, head_dim) tensor that the custom self-attention layer
    consumes via einsum.
    """

    def __init__(self, hidden_dim: int, num_heads: int, d_model: int,
                 sigma_d: float = 0.2, sigma_a: float = 15.0,
                 angle_k: int = 3, reduction: str = 'max'):
        super().__init__()
        assert reduction in ('max', 'mean')
        assert d_model % num_heads == 0, (
            f'd_model={d_model} must be divisible by num_heads={num_heads}')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.sigma_d = sigma_d
        self.factor_a = 180.0 / (sigma_a * math.pi)  # maps radians → ~O(1)
        self.angle_k = angle_k
        self.reduction = reduction

        self.embedding = SinusoidalScalarEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)
        # Project the (distance + angle) pair embedding into key space so it
        # can be dotted with per-head queries. Shape contract: produces an
        # R_ij vector in head_dim space for each (i, j) pair.
        self.proj_r = nn.Linear(hidden_dim, self.head_dim)

        # Zero-init the final relational projection so the pathway starts as
        # a NO-OP (R = 0 everywhere → Q·R = 0 → standard self-attention at
        # step 0). The model then gradually learns to use the relational
        # signal as training progresses. Without this, the untrained R
        # disrupts attention during cold-start optimisation — exactly the
        # failure mode we observed in the additive-bias version on the
        # `exp/replos` branch (rising val loss across early epochs).
        nn.init.zeros_(self.proj_r.weight)
        nn.init.zeros_(self.proj_r.bias)

    @torch.no_grad()
    def _distances_and_angles(self, points: torch.Tensor):
        # points: (N, 3)
        N = points.shape[0]
        diff = points.unsqueeze(0) - points.unsqueeze(1)  # (N, N, 3): v[i,j] = p_j - p_i
        dists = diff.norm(dim=-1)  # (N, N)

        k = min(self.angle_k + 1, N)
        knn_idx = dists.topk(k=k, largest=False, dim=-1).indices[..., 1:]  # (N, K) excl. self
        K = knn_idx.shape[-1]
        if K == 0:
            angles = torch.zeros(N, N, 1, device=points.device, dtype=points.dtype)
        else:
            neighbor_pts = points[knn_idx]  # (N, K, 3)
            v_ix = neighbor_pts - points.unsqueeze(1)  # (N, K, 3)
            v_ij = diff  # (N, N, 3)
            v_ij_n = F.normalize(v_ij, dim=-1).unsqueeze(2)  # (N, N, 1, 3)
            v_ix_n = F.normalize(v_ix, dim=-1).unsqueeze(1)  # (N, 1, K, 3)
            cos = (v_ij_n * v_ix_n).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)  # (N, N, K)
            angles = torch.acos(cos)
            angles = torch.nan_to_num(angles, nan=0.0)
        return dists, angles

    def _embed_pair(self, points: torch.Tensor) -> torch.Tensor:
        dists, angles = self._distances_and_angles(points)  # (N,N), (N,N,K)
        d_emb = self.proj_d(self.embedding(dists / self.sigma_d))  # (N, N, hidden)
        a_emb = self.proj_a(self.embedding(angles * self.factor_a))  # (N, N, K, hidden)
        if self.reduction == 'max':
            a_emb = a_emb.max(dim=-2).values
        else:
            a_emb = a_emb.mean(dim=-2)
        pair_emb = d_emb + a_emb  # (N, N, hidden)
        r = self.proj_r(pair_emb)  # (N, N, head_dim) — head-shared K-space embedding
        return r

    def forward(self, points_list):
        """points_list: list of (N_i, 3) — one per batch sample.
        Returns: list of (N_i, N_i, head_dim) — per-pair K-space relational
        embeddings to be combined with queries via the Q·R term in
        GeometricMultiheadAttention.
        """
        return [self._embed_pair(p) for p in points_list]


class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """
    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, n_dim: int = 1, d_model: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.mlp(xyz)