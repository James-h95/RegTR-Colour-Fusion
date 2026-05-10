"""Modified from DETR's transformer.py

- Cross encoder layer is similar to the decoder layers in Transformer, but
  updates both source and target features
- Added argument to control whether value has position embedding or not for
  TransformerEncoderLayer and TransformerDecoderLayer
- Decoder layer now keeps track of attention weights
"""

import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class GeometricMultiheadAttention(nn.Module):
    """Self-attention with optional GeoTransformer-style relational Q·R term.

    Standard scaled dot-product attention computes
        attn[b, h, i, j] = (Q[b, h, i, :] · K[b, h, j, :]) / sqrt(d).

    With a `relation` tensor R provided, the attention logits gain an
    additional content-aware geometric term:
        attn[b, h, i, j] = (Q[b, h, i, :] · K[b, h, j, :]
                            + Q[b, h, i, :] · R[b, i, j, :]) / sqrt(d).

    R is shared across heads (one (head_dim,) vector per pair (i, j))
    matching GeoTransformer's published formulation. The Q·R term is what
    distinguishes Tier 3 from the simplified additive-logit-bias port we
    used previously: here the geometric contribution is modulated by the
    query content, so different queries can attend to different geometric
    patterns instead of every query seeing the same scalar bias for a
    given pair.

    Mirrors nn.MultiheadAttention's call signature (seq-first tensors,
    optional key_padding_mask) for drop-in replacement at the SA call site.
    Cross-attention continues to use stock nn.MultiheadAttention because
    the two point clouds live in different frames and a "relative" geometric
    encoding between them isn't well-defined.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0, (
            f'd_model={d_model} must be divisible by nhead={nhead}')
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                relation: Optional[Tensor] = None):
        """
        Args:
          query, key, value: (N, B, d_model) — PyTorch seq-first convention.
          key_padding_mask: (B, Nk) bool — True positions are masked out.
          relation: optional (B, Nq, Nk, head_dim) — per-pair K-space
            geometric embedding for the Q·R term. Shared across heads.

        Returns:
          out: (Nq, B, d_model)
          attn_weights_avg: (B, Nq, Nk) attention weights averaged across
            heads (matching nn.MultiheadAttention's return convention so
            downstream `satt_weights` analysis code keeps working).
        """
        Nq, B, _ = query.shape
        Nk = key.shape[0]

        # Project & reshape to (B, H, N, D)
        Q = self.q_proj(query).view(Nq, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        K = self.k_proj(key).view(Nk, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)
        V = self.v_proj(value).view(Nk, B, self.nhead, self.head_dim).permute(1, 2, 0, 3)

        # Standard content-content logits
        attn = torch.matmul(Q, K.transpose(-1, -2))  # (B, H, Nq, Nk)

        # GeoTransformer Q·R term (what makes this Tier 3, not Tier 1).
        if relation is not None:
            # Q: (B, H, Nq, D); relation: (B, Nq, Nk, D); -> (B, H, Nq, Nk).
            r_logits = torch.einsum('bhid,bijd->bhij', Q, relation)
            attn = attn + r_logits

        attn = attn * self.scale

        if key_padding_mask is not None:
            # Broadcast (B, Nk) -> (B, 1, 1, Nk).
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float('-inf'))

        attn_weights = torch.softmax(attn, dim=-1)
        # If a row is fully masked the softmax produces NaN; fix to zeros so
        # the matmul below stays finite. (key_padding_mask covers a whole
        # row when query position itself is padded — that row's output gets
        # discarded downstream, but we still need finite values to avoid
        # propagating NaN through layer norm.)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, V)  # (B, H, Nq, D)
        out = out.permute(2, 0, 1, 3).contiguous().view(Nq, B, self.d_model)
        out = self.out_proj(out)

        return out, attn_weights.mean(dim=1)


class TransformerCrossEncoder(nn.Module):

    def __init__(self, cross_encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(cross_encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_sa_relation: Optional[Tensor] = None,
                tgt_sa_relation: Optional[Tensor] = None,):

        src_intermediate, tgt_intermediate = [], []

        for layer in self.layers:
            src, tgt = layer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                             src_key_padding_mask=src_key_padding_mask,
                             tgt_key_padding_mask=tgt_key_padding_mask,
                             src_pos=src_pos, tgt_pos=tgt_pos,
                             src_sa_relation=src_sa_relation,
                             tgt_sa_relation=tgt_sa_relation)
            if self.return_intermediate:
                src_intermediate.append(self.norm(src) if self.norm is not None else src)
                tgt_intermediate.append(self.norm(tgt) if self.norm is not None else tgt)

        if self.norm is not None:
            src = self.norm(src)
            tgt = self.norm(tgt)
            if self.return_intermediate:
                if len(self.layers) > 0:
                    src_intermediate.pop()
                    tgt_intermediate.pop()
                src_intermediate.append(src)
                tgt_intermediate.append(tgt)

        if self.return_intermediate:
            return torch.stack(src_intermediate), torch.stack(tgt_intermediate)

        return src.unsqueeze(0), tgt.unsqueeze(0)

    def get_attentions(self):
        """For analysis: Retrieves the attention maps last computed by the individual layers."""

        src_satt_all, tgt_satt_all = [], []
        src_xatt_all, tgt_xatt_all = [], []

        for layer in self.layers:
            src_satt, tgt_satt = layer.satt_weights
            src_xatt, tgt_xatt = layer.xatt_weights

            src_satt_all.append(src_satt)
            tgt_satt_all.append(tgt_satt)
            src_xatt_all.append(src_xatt)
            tgt_xatt_all.append(tgt_xatt)

        src_satt_all = torch.stack(src_satt_all)
        tgt_satt_all = torch.stack(tgt_satt_all)
        src_xatt_all = torch.stack(src_xatt_all)
        tgt_xatt_all = torch.stack(tgt_xatt_all)

        return (src_satt_all, tgt_satt_all), (src_xatt_all, tgt_xatt_all)


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 sa_val_has_pos_emb=False,
                 ca_val_has_pos_emb=False,
                 attention_type='dot_prod'
                 ):
        super().__init__()

        # Self-attention uses our custom GeometricMultiheadAttention so the
        # encoder can consume the (B, Nq, Nk, head_dim) relational tensor
        # produced by GeometricStructureEmbedding. Cross-attention stays as
        # stock nn.MultiheadAttention because src and tgt live in different
        # frames; relative geometry across the two clouds isn't well-defined.
        if attention_type == 'dot_prod':
            self.self_attn = GeometricMultiheadAttention(d_model, nhead, dropout=dropout)
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            raise NotImplementedError

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.sa_val_has_pos_emb = sa_val_has_pos_emb
        self.ca_val_has_pos_emb = ca_val_has_pos_emb
        self.satt_weights, self.xatt_weights = None, None  # For analysis

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, tgt,
                     src_mask: Optional[Tensor] = None,
                     tgt_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     src_pos: Optional[Tensor] = None,
                     tgt_pos: Optional[Tensor] = None,
                     src_sa_relation: Optional[Tensor] = None,
                     tgt_sa_relation: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention (Q·K + Q·R via GeometricMultiheadAttention)
        src_w_pos = self.with_pos_embed(src, src_pos)
        q = k = src_w_pos
        src2, satt_weights_s = self.self_attn(
            q, k,
            value=src_w_pos if self.sa_val_has_pos_emb else src,
            key_padding_mask=src_key_padding_mask,
            relation=src_sa_relation)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)
        q = k = tgt_w_pos
        tgt2, satt_weights_t = self.self_attn(
            q, k,
            value=tgt_w_pos if self.sa_val_has_pos_emb else tgt,
            key_padding_mask=tgt_key_padding_mask,
            relation=tgt_sa_relation)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention
        src_w_pos = self.with_pos_embed(src, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt, tgt_pos)

        src2, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt,
                                                   attn_mask=tgt_mask,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt2, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)

        src = self.norm2(src + self.dropout2(src2))
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        # Position-wise feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward_pre(self, src, tgt,
                    src_mask: Optional[Tensor] = None,
                    tgt_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    src_pos: Optional[Tensor] = None,
                    tgt_pos: Optional[Tensor] = None,
                    src_sa_relation: Optional[Tensor] = None,
                    tgt_sa_relation: Optional[Tensor] = None,):

        assert src_mask is None and tgt_mask is None, 'Masking not implemented'

        # Self attention (Q·K + Q·R via GeometricMultiheadAttention)
        src2 = self.norm1(src)
        src2_w_pos = self.with_pos_embed(src2, src_pos)
        q = k = src2_w_pos
        src2, satt_weights_s = self.self_attn(
            q, k,
            value=src2_w_pos if self.sa_val_has_pos_emb else src2,
            key_padding_mask=src_key_padding_mask,
            relation=src_sa_relation)
        src = src + self.dropout1(src2)

        tgt2 = self.norm1(tgt)
        tgt2_w_pos = self.with_pos_embed(tgt2, tgt_pos)
        q = k = tgt2_w_pos
        tgt2, satt_weights_t = self.self_attn(
            q, k,
            value=tgt2_w_pos if self.sa_val_has_pos_emb else tgt2,
            key_padding_mask=tgt_key_padding_mask,
            relation=tgt_sa_relation)
        tgt = tgt + self.dropout1(tgt2)

        # Cross attention
        src2, tgt2 = self.norm2(src), self.norm2(tgt)
        src_w_pos = self.with_pos_embed(src2, src_pos)
        tgt_w_pos = self.with_pos_embed(tgt2, tgt_pos)

        src3, xatt_weights_s = self.multihead_attn(query=self.with_pos_embed(src2, src_pos),
                                                   key=tgt_w_pos,
                                                   value=tgt_w_pos if self.ca_val_has_pos_emb else tgt2,
                                                   attn_mask=tgt_mask,
                                                   key_padding_mask=tgt_key_padding_mask)
        tgt3, xatt_weights_t = self.multihead_attn(query=self.with_pos_embed(tgt2, tgt_pos),
                                                   key=src_w_pos,
                                                   value=src_w_pos if self.ca_val_has_pos_emb else src2,
                                                   attn_mask=src_mask,
                                                   key_padding_mask=src_key_padding_mask)

        src = src + self.dropout2(src3)
        tgt = tgt + self.dropout2(tgt3)

        # Position-wise feedforward
        src2 = self.norm3(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        # Stores the attention weights for analysis, if required
        self.satt_weights = (satt_weights_s, satt_weights_t)
        self.xatt_weights = (xatt_weights_s, xatt_weights_t)

        return src, tgt

    def forward(self, src, tgt,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                src_pos: Optional[Tensor] = None,
                tgt_pos: Optional[Tensor] = None,
                src_sa_relation: Optional[Tensor] = None,
                tgt_sa_relation: Optional[Tensor] = None,):

        if self.normalize_before:
            return self.forward_pre(src, tgt, src_mask, tgt_mask,
                                    src_key_padding_mask, tgt_key_padding_mask,
                                    src_pos, tgt_pos,
                                    src_sa_relation, tgt_sa_relation)
        return self.forward_post(src, tgt, src_mask, tgt_mask,
                                 src_key_padding_mask, tgt_key_padding_mask,
                                 src_pos, tgt_pos,
                                 src_sa_relation, tgt_sa_relation)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

