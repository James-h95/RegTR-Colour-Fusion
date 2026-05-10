"""REGTR network architecture
"""
import math

import torch
import torch.nn as nn

from models.backbone_kpconv.kpconv import KPFEncoder, PreprocessorGPU, compute_overlaps
from models.generic_reg_model import GenericRegModel
from models.losses.corr_loss import CorrCriterion
from models.losses.feature_loss import InfoNCELossFull, CircleLossFull
from models.transformer.position_embedding import PositionEmbeddingCoordsSine, \
    PositionEmbeddingLearned, GeometricStructureEmbedding
from models.transformer.transformers import \
    TransformerCrossEncoderLayer, TransformerCrossEncoder
from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_inv
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences
from utils.viz import visualize_registration
_TIMEIT = False
# One-shot wiring diagnostic: logs a single line on the very first forward
# pass confirming that (a) RGB actually reaches feats0 when use_color=True
# (i.e. it's not silently falling through to ones), and (b) the relational
# embedding tensor has a sane magnitude when use_relational_pos_emb=True.
# This is the fingerprint you grep for to distinguish "my change is a no-op"
# from "my change is active but undertrained" in ablation logs.
_WIRING_LOGGED = False


def _pack_sa_relation(rel_list):
    """Pad a list of per-sample (N_i, N_i, head_dim) relational embeddings
    into a single (B, N_max, N_max, head_dim) tensor for the Q·R term in
    GeometricMultiheadAttention.

    Padding is zero — for any (i, j) where either i or j is a padded
    position the relation contributes nothing, and the corresponding
    attention logits are masked out by `key_padding_mask` in the encoder
    layer anyway.
    """
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


class RegTR(GenericRegModel):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

        #######################
        # Preprocessor
        #######################
        self.preprocessor = PreprocessorGPU(cfg)

        #######################
        # KPConv Encoder/decoder
        #######################
        self.kpf_encoder = KPFEncoder(cfg, cfg.d_embed)
        # Bottleneck layer to shrink KPConv features to a smaller dimension for running attention
        self.feat_proj = nn.Linear(self.kpf_encoder.encoder_skip_dims[-1], cfg.d_embed, bias=True)

        #######################
        # Embeddings
        #######################
        if cfg.get('pos_emb_type', 'sine') == 'sine':
            self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.d_embed,
                                                         scale=cfg.get('pos_emb_scaling', 1.0))
        elif cfg['pos_emb_type'] == 'learned':
            self.pos_embed = PositionEmbeddingLearned(3, cfg.d_embed)
        else:
            raise NotImplementedError

        # Colour fusion (Task 1, CEFE-1). When enabled, feats0 is the per-point
        # RGB (3 channels) instead of ones, so in_feats_dim must be 3.
        # load_config flattens kpconv_options into cfg, so in_feats_dim is
        # accessed at the top level.
        self.use_color = bool(cfg.get('use_color', False))
        if self.use_color:
            assert cfg.in_feats_dim == 3, (
                f'use_color=True requires kpconv_options.in_feats_dim: 3 in the '
                f'config, got {cfg.in_feats_dim}.'
            )

        # GeoTransformer-style relational encoding for self-attention (Tier 3).
        # When enabled, the encoder's self-attention computes
        #     attn[i,j] = (Q_i · K_j + Q_i · R_ij) / sqrt(d)
        # where R_ij is a head-shared per-pair embedding in K-space (head_dim).
        # Cross-attention is unchanged. Absolute sine pos_embed (above) can
        # still be enabled in the config and used additively with the
        # relational signal — the previous failure mode of pure replacement
        # is captured on the `exp/replos` branch.
        self.use_relational_pos_emb = cfg.get('use_relational_pos_emb', False)
        if self.use_relational_pos_emb:
            self.geo_embed = GeometricStructureEmbedding(
                hidden_dim=cfg.get('relational_hidden_dim', 128),
                num_heads=cfg.nhead,
                d_model=cfg.d_embed,
                sigma_d=cfg.get('relational_sigma_d', 0.2),
                sigma_a=cfg.get('relational_sigma_a', 15.0),
                angle_k=cfg.get('relational_angle_k', 3),
            )

        #######################
        # Attention propagation
        #######################
        encoder_layer = TransformerCrossEncoderLayer(
            cfg.d_embed, cfg.nhead, cfg.d_feedforward, cfg.dropout,
            activation=cfg.transformer_act,
            normalize_before=cfg.pre_norm,
            sa_val_has_pos_emb=cfg.sa_val_has_pos_emb,
            ca_val_has_pos_emb=cfg.ca_val_has_pos_emb,
            attention_type=cfg.attention_type,
        )
        encoder_norm = nn.LayerNorm(cfg.d_embed) if cfg.pre_norm else None
        self.transformer_encoder = TransformerCrossEncoder(
            encoder_layer, cfg.num_encoder_layers, encoder_norm,
            return_intermediate=True)

        #######################
        # Output layers
        #######################
        if cfg.get('direct_regress_coor', False):
            self.correspondence_decoder = CorrespondenceRegressor(cfg.d_embed)
        else:
            self.correspondence_decoder = CorrespondenceDecoder(cfg.d_embed,
                                                                cfg.corr_decoder_has_pos_emb,
                                                                self.pos_embed)

        #######################
        # Losses
        #######################
        self.overlap_criterion = nn.BCEWithLogitsLoss()
        if self.cfg.feature_loss_type == 'infonce':
            self.feature_criterion = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
        elif self.cfg.feature_loss_type == 'circle':
            self.feature_criterion = CircleLossFull(dist_type='euclidean', r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = self.feature_criterion
        else:
            raise NotImplementedError

        self.corr_criterion = CorrCriterion(metric='mae')

        self.weight_dict = {}
        for k in ['overlap', 'feature', 'corr']:
            for i in cfg.get(f'{k}_loss_on', [cfg.num_encoder_layers - 1]):
                self.weight_dict[f'{k}_{i}'] = cfg.get(f'wt_{k}')
        self.weight_dict['feature_un'] = cfg.wt_feature_un

        self.logger.info('Loss weighting: {}'.format(self.weight_dict))
        self.logger.info(
            f'Config: d_embed:{cfg.d_embed}, nheads:{cfg.nhead}, pre_norm:{cfg.pre_norm}, '
            f'use_pos_emb:{cfg.transformer_encoder_has_pos_emb}, '
            f'sa_val_has_pos_emb:{cfg.sa_val_has_pos_emb}, '
            f'ca_val_has_pos_emb:{cfg.ca_val_has_pos_emb}'
        )

    def forward(self, batch):
        B = len(batch['src_xyz'])
        outputs = {}

        if _TIMEIT:
            t_start_all_cuda, t_end_all_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_pp_cuda, t_end_pp_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_all_cuda.record()
            t_start_pp_cuda.record()

        # Preprocess
        kpconv_meta = self.preprocessor(batch['src_xyz'] + batch['tgt_xyz'])
        batch['kpconv_meta'] = kpconv_meta
        slens = [s.tolist() for s in kpconv_meta['stack_lengths']]
        slens_c = slens[-1]
        src_slens_c, tgt_slens_c = slens_c[:B], slens_c[B:]

        # Initial features for the KPConv encoder.
        # Baseline REGTR: geometry-blind ones (in_feats_dim=1).
        # Task 1 (CEFE-1): per-point RGB injected as initial features. We follow
        # the preprocessor's src-then-tgt stacking order so feats line up with
        # kpconv_meta['points'][0].
        pts0 = kpconv_meta['points'][0]
        if self.use_color and 'src_rgb' in batch and 'tgt_rgb' in batch:
            rgb_stacked = torch.cat(
                [r.to(pts0.device) for r in batch['src_rgb'] + batch['tgt_rgb']],
                dim=0,
            )
            if rgb_stacked.shape[0] != pts0.shape[0]:
                # Include per-fragment shapes so the log points at the
                # offending item instead of just the stacked totals.
                src_shapes = [tuple(x.shape) for x in batch['src_xyz']]
                tgt_shapes = [tuple(x.shape) for x in batch['tgt_xyz']]
                src_rgb_shapes = [tuple(x.shape) for x in batch['src_rgb']]
                tgt_rgb_shapes = [tuple(x.shape) for x in batch['tgt_rgb']]
                raise AssertionError(
                    f'RGB/point count mismatch: {rgb_stacked.shape[0]} vs '
                    f'{pts0.shape[0]}. src_xyz={src_shapes}, '
                    f'tgt_xyz={tgt_shapes}, src_rgb={src_rgb_shapes}, '
                    f'tgt_rgb={tgt_rgb_shapes}')
            # Zero-center RGB ([0, 1] -> [-0.5, 0.5]) before feeding it to
            # the first KPConv layer. Baseline REGTR's `feats0 = ones`
            # input has a fixed scale that the kernels are implicitly
            # calibrated for; raw RGB has mean ~0.5 and the resulting
            # asymmetric activations occasionally produce extreme single-
            # kernel responses on certain fragments, blowing up the
            # gradient w.r.t. kpf_encoder.encoder_blocks.0.KPConv.weights.
            # Centering halves the worst-case input magnitude and removes
            # the systematic positive bias without losing any colour
            # information.
            feats0 = rgb_stacked - 0.5
        else:
            feats0 = torch.ones_like(pts0[:, 0:1])

        if _TIMEIT:
            t_end_pp_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_pp_cuda = t_start_pp_cuda.elapsed_time(t_end_pp_cuda) / 1000
            t_start_enc_cuda, t_end_enc_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_enc_cuda.record()

        ####################
        # REGTR Encoder
        ####################
        # KPConv encoder (downsampling) to obtain unconditioned features
        feats_un, skip_x = self.kpf_encoder(feats0, kpconv_meta)
        if _TIMEIT:
            t_end_enc_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_enc_cuda = t_start_enc_cuda.elapsed_time(t_end_enc_cuda) / 1000
            t_start_att_cuda, t_end_att_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_att_cuda.record()

        both_feats_un = self.feat_proj(feats_un)
        src_feats_un, tgt_feats_un = split_src_tgt(both_feats_un, slens_c)

        # Position embedding for downsampled points
        src_xyz_c, tgt_xyz_c = split_src_tgt(kpconv_meta['points'][-1], slens_c)
        src_pe, tgt_pe = split_src_tgt(self.pos_embed(kpconv_meta['points'][-1]), slens_c)
        src_pe_padded, _, _ = pad_sequence(src_pe)
        tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

        # GeoTransformer-style relational embedding for self-attention (Tier 3).
        # Per-pair head-shared K-space tensor (B, N_max, N_max, head_dim) that
        # the encoder layer adds as a Q·R term to its attention logits.
        src_sa_relation = tgt_sa_relation = None
        if self.use_relational_pos_emb:
            src_sa_relation = _pack_sa_relation(self.geo_embed(src_xyz_c))
            tgt_sa_relation = _pack_sa_relation(self.geo_embed(tgt_xyz_c))

        # One-shot wiring diagnostic (see module-level comment). Logs a
        # single line on the first forward pass of a run. If use_color=True
        # but feats0 is exactly ones, RGB never reached the model. With
        # zero-init on proj_r, src_sa_relation should report absmean ~0 at
        # step 0 and grow as training progresses; non-zero at step 0 means
        # zero-init didn't take.
        global _WIRING_LOGGED
        if not _WIRING_LOGGED:
            with torch.no_grad():
                f = feats0
                msg = (
                    f'[WIRING] use_color={self.use_color} '
                    f'use_relational_pos_emb={self.use_relational_pos_emb} '
                    f'| feats0: shape={tuple(f.shape)} '
                    f'mean={f.mean().item():.4f} std={f.std().item():.4f} '
                    f'min={f.min().item():.4f} max={f.max().item():.4f}'
                )
                if src_sa_relation is not None:
                    r = src_sa_relation
                    msg += (
                        f' | src_sa_relation: shape={tuple(r.shape)} '
                        f'mean={r.mean().item():.4f} std={r.std().item():.4f} '
                        f'absmean={r.abs().mean().item():.4f} '
                        f'min={r.min().item():.4f} max={r.max().item():.4f}'
                    )
            self.logger.info(msg)
            _WIRING_LOGGED = True

        # Performs padding, then apply attention (REGTR "encoder" stage) to condition on the other
        # point cloud
        src_feats_padded, src_key_padding_mask, _ = pad_sequence(src_feats_un,
                                                                 require_padding_mask=True)
        tgt_feats_padded, tgt_key_padding_mask, _ = pad_sequence(tgt_feats_un,
                                                                 require_padding_mask=True)
        src_feats_cond, tgt_feats_cond = self.transformer_encoder(
            src_feats_padded, tgt_feats_padded,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_pos=src_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
            tgt_pos=tgt_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
            src_sa_relation=src_sa_relation,
            tgt_sa_relation=tgt_sa_relation,
        )

        src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list = \
            self.correspondence_decoder(src_feats_cond, tgt_feats_cond, src_xyz_c, tgt_xyz_c)

        src_feats_list = unpad_sequences(src_feats_cond, src_slens_c)
        tgt_feats_list = unpad_sequences(tgt_feats_cond, tgt_slens_c)
        num_pred = src_feats_cond.shape[0]

        ## TIMING CODE
        if _TIMEIT:
            t_end_att_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_att_cuda = t_start_att_cuda.elapsed_time(t_end_att_cuda) / 1000
            t_start_pose_cuda, t_end_pose_cuda = \
                torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t_start_pose_cuda.record()

        # Stacks correspondences in both directions and computes the pose
        corr_all, overlap_prob = [], []
        for b in range(B):
            corr_all.append(torch.cat([
                torch.cat([src_xyz_c[b].expand(num_pred, -1, -1), src_corr_list[b]], dim=2),
                torch.cat([tgt_corr_list[b], tgt_xyz_c[b].expand(num_pred, -1, -1)], dim=2)
            ], dim=1))
            # Sigmoid maps to [0, 1] for any finite input, but NaN/Inf
            # overlap logits (from a numerically unstable forward pass)
            # slip through and later fail the weighted Kabsch sanity
            # check at se3_torch.py:125. Treat non-finite weights as
            # "zero confidence" so the batch degrades gracefully instead
            # of aborting; non-finite-loss detection in the trainer still
            # stops corrupt gradients from reaching the optimizer.
            overlap_prob_b = torch.cat([
                torch.sigmoid(src_overlap_list[b][:, :, 0]),
                torch.sigmoid(tgt_overlap_list[b][:, :, 0]),
            ], dim=1)
            overlap_prob_b = torch.nan_to_num(
                overlap_prob_b, nan=0.0, posinf=1.0, neginf=0.0
            ).clamp_(0.0, 1.0)
            overlap_prob.append(overlap_prob_b)

            # # Thresholds the overlap probability. Enable this for inference to get a slight boost
            # # in performance. However, we do not use this in the paper.
            # overlap_prob = [nn.functional.threshold(overlap_prob[b], 0.5, 0.0) for b in range(B)]

        pred_pose_weighted = torch.stack([
            compute_rigid_transform(corr_all[b][..., :3], corr_all[b][..., 3:],
                                    overlap_prob[b])
            for b in range(B)], dim=1)

        ## TIMING CODE
        if _TIMEIT:
            t_end_pose_cuda.record()
            t_end_all_cuda.record()
            torch.cuda.synchronize()
            t_elapsed_pose_cuda = t_start_pose_cuda.elapsed_time(t_end_pose_cuda) / 1000
            t_elapsed_all_cuda = t_start_all_cuda.elapsed_time(t_end_all_cuda) / 1000
            with open('timings.txt', 'a') as fid:
                fid.write('{:10f}\t{:10f}\t{:10f}\t{:10f}\t{:10f}\n'.format(
                    t_elapsed_pp_cuda, t_elapsed_enc_cuda, t_elapsed_att_cuda,
                    t_elapsed_pose_cuda, t_elapsed_all_cuda
                ))

        outputs = {
            # Predictions
            'src_feat_un': src_feats_un,
            'tgt_feat_un': tgt_feats_un,
            'src_feat': src_feats_list,  # List(B) of (N_pred, N_src, D)
            'tgt_feat': tgt_feats_list,  # List(B) of (N_pred, N_tgt, D)

            'src_kp': src_xyz_c,
            'src_kp_warped': src_corr_list,
            'tgt_kp': tgt_xyz_c,
            'tgt_kp_warped': tgt_corr_list,

            'src_overlap': src_overlap_list,
            'tgt_overlap': tgt_overlap_list,

            'pose': pred_pose_weighted,
        }
        return outputs

    def compute_loss(self, pred, batch):
        losses = {}
        kpconv_meta = batch['kpconv_meta']
        pose_gt = batch['pose']
        p = len(kpconv_meta['stack_lengths']) - 1  # coarsest level

        # Compute groundtruth overlaps first
        batch['overlap_pyr'] = compute_overlaps(batch)
        src_overlap_p, tgt_overlap_p = \
            split_src_tgt(batch['overlap_pyr'][f'pyr_{p}'], kpconv_meta['stack_lengths'][p])

        # Overlap prediction loss
        all_overlap_pred = torch.cat(pred['src_overlap'] + pred['tgt_overlap'], dim=-2)
        all_overlap_gt = batch['overlap_pyr'][f'pyr_{p}']
        for i in self.cfg.overlap_loss_on:
            losses[f'overlap_{i}'] = self.overlap_criterion(all_overlap_pred[i, :, 0], all_overlap_gt)

        # Feature criterion
        for i in self.cfg.feature_loss_on:
            losses[f'feature_{i}'] = self.feature_criterion(
                [s[i] for s in pred['src_feat']],
                [t[i] for t in pred['tgt_feat']],
                se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
            )
        losses['feature_un'] = self.feature_criterion_un(
            pred['src_feat_un'],
            pred['tgt_feat_un'],
            se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
        )

        # Loss on the 6D correspondences
        for i in self.cfg.corr_loss_on:
            src_corr_loss = self.corr_criterion(
                pred['src_kp'],
                [w[i] for w in pred['src_kp_warped']],
                batch['pose'],
                overlap_weights=src_overlap_p
            )
            tgt_corr_loss = self.corr_criterion(
                pred['tgt_kp'],
                [w[i] for w in pred['tgt_kp_warped']],
                torch.stack([se3_inv(p) for p in batch['pose']]),
                overlap_weights=tgt_overlap_p
            )
            losses[f'corr_{i}'] = src_corr_loss + tgt_corr_loss

        debug = False  # Set this to true to look at the registration result
        if debug:
            b = 0
            o = -1  # Visualize output of final transformer layer
            visualize_registration(batch['src_xyz'][b], batch['tgt_xyz'][b],
                                   torch.cat([pred['src_kp'][b], pred['src_kp_warped'][b][o]], dim=1),
                                   correspondence_conf=torch.sigmoid(pred['src_overlap'][b][o])[:, 0],
                                   pose_gt=pose_gt[b], pose_pred=pred['pose'][o, b])

        losses['total'] = torch.sum(
            torch.stack([(losses[k] * self.weight_dict[k]) for k in losses]))
        return losses


class CorrespondenceDecoder(nn.Module):
    def __init__(self, d_embed, use_pos_emb, pos_embed=None, num_neighbors=0):
        super().__init__()

        assert use_pos_emb is False or pos_embed is not None, \
            'Position encoder must be supplied if use_pos_emb is True'

        self.use_pos_emb = use_pos_emb
        self.pos_embed = pos_embed
        self.q_norm = nn.LayerNorm(d_embed)

        self.q_proj = nn.Linear(d_embed, d_embed)
        self.k_proj = nn.Linear(d_embed, d_embed)
        self.conf_logits_decoder = nn.Linear(d_embed, 1)
        self.num_neighbors = num_neighbors

        # nn.init.xavier_uniform_(self.q_proj.weight)
        # nn.init.xavier_uniform_(self.k_proj.weight)

    def simple_attention(self, query, key, value, key_padding_mask=None):
        """Simplified single-head attention that does not project the value:
        Linearly projects only the query and key, compute softmax dot product
        attention, then returns the weighted sum of the values

        Args:
            query: ([N_pred,] Q, B, D)
            key: ([N_pred,] S, B, D)
            value: (S, B, E), i.e. dimensionality can be different
            key_padding_mask: (B, S)

        Returns:
            Weighted values (B, Q, E)
        """

        q = self.q_proj(query) / math.sqrt(query.shape[-1])
        k = self.k_proj(key)

        attn = torch.einsum('...qbd,...sbd->...bqs', q, k)  # (B, N_query, N_src)

        if key_padding_mask is not None:
            attn_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
            attn_mask.masked_fill_(key_padding_mask, float('-inf'))
            attn = attn + attn_mask[:, None, :]  # ([N_pred,] B, Q, S)

        if self.num_neighbors > 0:
            neighbor_mask = torch.full_like(attn, fill_value=float('-inf'))
            haha = torch.topk(attn, k=self.num_neighbors, dim=-1).indices
            neighbor_mask[:, :, haha] = 0
            attn = attn + neighbor_mask

        attn = torch.softmax(attn, dim=-1)

        attn_out = torch.einsum('...bqs,...sbd->...qbd', attn, value)

        return attn_out

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3)
            tgt_xyz: List of ([N_pred,] N_tgt, 3)

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)
        assert src_xyz_padded.shape[:-1] == src_feats_padded.shape[-3:-1] and \
               tgt_xyz_padded.shape[:-1] == tgt_feats_padded.shape[-3:-1]

        if self.use_pos_emb:
            both_xyz_packed = torch.cat(src_xyz + tgt_xyz)
            slens = list(map(len, src_xyz)) + list(map(len, tgt_xyz))
            src_pe, tgt_pe = split_src_tgt(self.pos_embed(both_xyz_packed), slens)
            src_pe_padded, _, _ = pad_sequence(src_pe)
            tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

        # Decode the coordinates
        src_feats2 = src_feats_padded + src_pe_padded if self.use_pos_emb else src_feats_padded
        tgt_feats2 = tgt_feats_padded + tgt_pe_padded if self.use_pos_emb else tgt_feats_padded
        src_corr = self.simple_attention(src_feats2, tgt_feats2, pad_sequence(tgt_xyz)[0],
                                         tgt_key_padding_mask)
        tgt_corr = self.simple_attention(tgt_feats2, src_feats2, pad_sequence(src_xyz)[0],
                                         src_key_padding_mask)

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list


class CorrespondenceRegressor(nn.Module):

    def __init__(self, d_embed):
        super().__init__()

        self.coor_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 3)
        )
        self.conf_logits_decoder = nn.Linear(d_embed, 1)

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3). Ignored
            tgt_xyz: List of ([N_pred,] N_tgt, 3). Ignored

        Returns:

        """

        src_xyz_padded, src_key_padding_mask, src_lens = \
            pad_sequence(src_xyz, require_padding_mask=True, require_lens=True)
        tgt_xyz_padded, tgt_key_padding_mask, tgt_lens = \
            pad_sequence(tgt_xyz, require_padding_mask=True, require_lens=True)

        # Decode the coordinates
        src_corr = self.coor_mlp(src_feats_padded)
        tgt_corr = self.coor_mlp(tgt_feats_padded)

        src_overlap = self.conf_logits_decoder(src_feats_padded)
        tgt_overlap = self.conf_logits_decoder(tgt_feats_padded)

        src_corr_list = unpad_sequences(src_corr, src_lens)
        tgt_corr_list = unpad_sequences(tgt_corr, tgt_lens)
        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        return src_corr_list, tgt_corr_list, src_overlap_list, tgt_overlap_list
