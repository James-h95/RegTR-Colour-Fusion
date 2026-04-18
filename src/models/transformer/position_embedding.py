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
    """GeoTransformer-style relational positional encoding.

    Produces a per-pair additive bias for self-attention that is invariant to
    rigid transforms of the input point cloud. Built from:
      * Pairwise distance embeddings (PDE).
      * Triplet-angle embeddings (TAE) aggregated over the k-nearest
        neighbours of each anchor.

    forward(points_list) returns a list of tensors shaped (num_heads, N_i, N_i)
    — one per sample — suitable for being stacked/padded into an attn_mask for
    nn.MultiheadAttention.
    """

    def __init__(self, hidden_dim: int, num_heads: int,
                 sigma_d: float = 0.2, sigma_a: float = 15.0,
                 angle_k: int = 3, reduction: str = 'max'):
        super().__init__()
        assert reduction in ('max', 'mean')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.sigma_d = sigma_d
        self.factor_a = 180.0 / (sigma_a * math.pi)  # maps radians → ~O(1)
        self.angle_k = angle_k
        self.reduction = reduction

        self.embedding = SinusoidalScalarEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, hidden_dim)
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)
        self.proj_bias = nn.Linear(hidden_dim, num_heads)

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
        d_emb = self.proj_d(self.embedding(dists / self.sigma_d))  # (N, N, H)
        a_emb = self.proj_a(self.embedding(angles * self.factor_a))  # (N, N, K, H)
        if self.reduction == 'max':
            a_emb = a_emb.max(dim=-2).values
        else:
            a_emb = a_emb.mean(dim=-2)
        pair_emb = d_emb + a_emb  # (N, N, H)
        bias = self.proj_bias(pair_emb)  # (N, N, num_heads)
        return bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N)

    def forward(self, points_list):
        """points_list: list of (N_i, 3) — one per batch sample."""
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