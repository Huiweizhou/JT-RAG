from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ANESConfig:
    sem_dim: int = 768
    str_dim: int = 256
    tmp_dim: int = 10
    meta_dim: int = 13
    hidden_dim: int = 256
    num_heads: int = 4
    set_attn_layers: int = 1
    dropout: float = 0.1
    max_time_points: int = 32


    use_pair_query_tokens: bool = False
    use_candidate_query_cross_attn: bool = True
    use_side_embedding: bool = True
    use_enhanced_stop: bool = True


    candidate_query_attn_chunk_size: int = 2048


class ModalProjection(nn.Module):


    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.norm(F.gelu(self.proj(x))))


class MLPScorer(nn.Module):


    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ANESSelector(nn.Module):


    def __init__(self, cfg: ANESConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim


        self.sem_proj = ModalProjection(cfg.sem_dim, d, cfg.dropout)
        self.str_proj = ModalProjection(cfg.str_dim, d, cfg.dropout)
        self.tmp_proj = ModalProjection(cfg.tmp_dim, d, cfg.dropout)
        self.meta_proj = ModalProjection(cfg.meta_dim, d, cfg.dropout)


        self.q_sem_proj = ModalProjection(cfg.sem_dim, d, cfg.dropout)
        self.q_str_proj = ModalProjection(cfg.str_dim, d, cfg.dropout)
        self.time_proj = ModalProjection(4, d, cfg.dropout)
        self.query_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.query_attn = nn.MultiheadAttention(d, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(d)


        self.candidate_query_attn = nn.MultiheadAttention(d, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.candidate_query_norm = nn.LayerNorm(d)


        self.mod_q = nn.Linear(d, d, bias=False)
        self.mod_k = nn.Linear(d, d, bias=False)
        self.mod_v = nn.Linear(d, d, bias=False)
        self.modality_bias = nn.Parameter(torch.zeros(4))
        self.modality_dropout = nn.Dropout(cfg.dropout)


        self.side_embedding = nn.Embedding(4, d)


        self.stop_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.num_heads,
            dim_feedforward=d * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.set_attn_layers)


        self.score_node = nn.Linear(d, d, bias=False)
        self.score_query = nn.Linear(d, d, bias=False)

        self.node_mlp = MLPScorer(d * 2, d, cfg.dropout)
        self.stop_mlp = MLPScorer(d * 3 + 2, d, cfg.dropout)

    def _time_features(self, sample_time: torch.Tensor, context_time: torch.Tensor) -> torch.Tensor:

        sample_time = sample_time.float()
        context_time = context_time.float()
        t_norm = (sample_time - 1944.0) / 100.0
        ctx_norm = (context_time - 1944.0) / 100.0
        gap_norm = (sample_time - context_time) / 20.0
        sin_t = torch.sin(t_norm * math.pi)
        return torch.stack([t_norm, ctx_norm, gap_norm, sin_t], dim=-1)

    def build_query_tokens(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        src_sem = batch["src_sem"]
        dst_sem = batch["dst_sem"]
        src_str = batch["src_str"]
        dst_str = batch["dst_str"]

        tokens = [
            self.q_sem_proj(src_sem),
            self.q_sem_proj(dst_sem),
            self.q_str_proj(src_str),
            self.q_str_proj(dst_str),
            self.time_proj(self._time_features(batch["sample_time"], batch["context_time"])),
        ]


        return torch.stack(tokens, dim=1)

    def build_query(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

        q_tokens = self.build_query_tokens(batch)
        bsz = q_tokens.size(0)
        q = self.query_token.expand(bsz, -1, -1)
        pooled, _ = self.query_attn(query=q, key=q_tokens, value=q_tokens, need_weights=False)
        return self.query_norm(pooled.squeeze(1))

    def _side_index(self, candidate_meta: torch.Tensor) -> torch.Tensor:

        if candidate_meta.size(-1) < 6:
            return torch.zeros(candidate_meta.shape[:-1], dtype=torch.long, device=candidate_meta.device)
        src_only = candidate_meta[..., 3] > 0.5
        dst_only = candidate_meta[..., 4] > 0.5
        common = candidate_meta[..., 5] > 0.5
        side = torch.zeros(candidate_meta.shape[:-1], dtype=torch.long, device=candidate_meta.device)
        side = torch.where(src_only, torch.ones_like(side), side)
        side = torch.where(dst_only, torch.full_like(side, 2), side)
        side = torch.where(common, torch.full_like(side, 3), side)
        return side

    def _query_cross_enhance(self, h_mod: torch.Tensor, q_tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        if not bool(self.cfg.use_candidate_query_cross_attn):
            return h_mod

        bsz, max_c, n_mod, d = h_mod.shape
        if max_c == 0:
            return h_mod

        flat_size = bsz * max_c
        q_len = q_tokens.size(1)
        chunk_size = int(getattr(self.cfg, "candidate_query_attn_chunk_size", 2048))
        chunk_size = max(1, chunk_size)

        q_rep = q_tokens.unsqueeze(1).expand(-1, max_c, -1, -1).reshape(flat_size, q_len, d)
        mod_in = h_mod.reshape(flat_size, n_mod, d)
        mod_out = torch.empty_like(mod_in)

        for start in range(0, flat_size, chunk_size):
            end = min(start + chunk_size, flat_size)
            attn_out, _ = self.candidate_query_attn(
                query=mod_in[start:end],
                key=q_rep[start:end],
                value=q_rep[start:end],
                need_weights=False,
            )
            mod_out[start:end] = self.candidate_query_norm(mod_in[start:end] + attn_out)

        mod_out = mod_out.reshape(bsz, max_c, n_mod, d)
        return torch.where(mask.unsqueeze(-1).unsqueeze(-1), mod_out, h_mod)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        mask = batch["candidate_mask"]
        bsz, max_c = mask.shape
        q_tokens = self.build_query_tokens(batch)
        q = self.query_token.expand(bsz, -1, -1)
        pooled, _ = self.query_attn(query=q, key=q_tokens, value=q_tokens, need_weights=False)
        hq = self.query_norm(pooled.squeeze(1))

        h_sem = self.sem_proj(batch["candidate_sem"])
        h_str = self.str_proj(batch["candidate_str"])
        h_tmp = self.tmp_proj(batch["candidate_tmp"])
        h_meta = self.meta_proj(batch["candidate_meta"])
        h_mod = torch.stack([h_sem, h_str, h_tmp, h_meta], dim=2)
        h_mod = self._query_cross_enhance(h_mod, q_tokens=q_tokens, mask=mask)

        q_vec = self.mod_q(hq).unsqueeze(1).unsqueeze(2)
        k_vec = self.mod_k(h_mod)
        mod_logits = (q_vec * k_vec).sum(-1) / math.sqrt(self.cfg.hidden_dim)
        mod_logits = mod_logits + self.modality_bias.view(1, 1, 4)
        alpha = torch.softmax(mod_logits, dim=-1)
        alpha = self.modality_dropout(alpha)
        v_vec = self.mod_v(h_mod)
        z = (alpha.unsqueeze(-1) * v_vec).sum(dim=2)

        if bool(self.cfg.use_side_embedding):
            side = self._side_index(batch["candidate_meta"])
            z = z + self.side_embedding(side)


        stop = self.stop_token.expand(bsz, 1, -1)
        set_in = torch.cat([z, stop], dim=1)
        stop_mask = torch.zeros((bsz, 1), dtype=torch.bool, device=mask.device)
        key_padding_mask = torch.cat([~mask, stop_mask], dim=1)
        set_out = self.set_encoder(set_in, src_key_padding_mask=key_padding_mask)
        cand_out = set_out[:, :max_c, :]
        stop_out = set_out[:, max_c, :]

        q_score = self.score_query(hq)
        bilinear_scores = (self.score_node(cand_out) * q_score.unsqueeze(1)).sum(-1) / math.sqrt(self.cfg.hidden_dim)
        mlp_in = torch.cat([
            cand_out,
            hq.unsqueeze(1).expand(-1, max_c, -1),
        ], dim=-1)
        cand_scores = bilinear_scores + self.node_mlp(mlp_in)
        cand_scores = cand_scores.masked_fill(~mask, -1e9)


        if bool(self.cfg.use_enhanced_stop):


            if max_c == 0:
                mean_pool = torch.zeros((bsz, self.cfg.hidden_dim), dtype=stop_out.dtype, device=stop_out.device)
                max_score = torch.zeros((bsz,), dtype=stop_out.dtype, device=stop_out.device)
                mean_score = torch.zeros((bsz,), dtype=stop_out.dtype, device=stop_out.device)
            else:
                mask_f = mask.float().unsqueeze(-1)
                valid_count = mask_f.sum(dim=1).clamp_min(1.0)
                mean_pool = (cand_out * mask_f) / valid_count.unsqueeze(-1)
                mean_pool = mean_pool.sum(dim=1)

                masked_scores = cand_scores.masked_fill(~mask, -1e9)
                max_score = masked_scores.max(dim=1).values

                has_candidate = mask.any(dim=1)
                max_score = torch.where(has_candidate, max_score, torch.zeros_like(max_score))
                mean_score = cand_scores.masked_fill(~mask, 0.0).sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)
                mean_score = torch.where(has_candidate, mean_score, torch.zeros_like(mean_score))

            stop_in = torch.cat([stop_out, hq, mean_pool, max_score.unsqueeze(-1), mean_score.unsqueeze(-1)], dim=-1)
            stop_score = self.stop_mlp(stop_in)
        else:
            stop_score = (self.score_node(stop_out) * q_score).sum(-1) / math.sqrt(self.cfg.hidden_dim)

        logits = cand_scores - stop_score.unsqueeze(1)
        probs = torch.sigmoid(logits).masked_fill(~mask, 0.0)

        return {
            "scores": cand_scores,
            "stop_score": stop_score,
            "probs": probs,
            "logits": logits.masked_fill(~mask, -1e9),
            "modality_attention": alpha,
            "hq": hq,
            "z": cand_out,
        }

    @torch.no_grad()
    def select_evidence(
        self,
        batch: Dict[str, torch.Tensor],
        k_max: int = 8,
        min_select: int = 0,
    ) -> Dict[str, torch.Tensor]:

        out = self.forward(batch)
        scores = out["scores"]
        stop = out["stop_score"].unsqueeze(1)
        mask = batch["candidate_mask"]
        selected_mask = (scores > stop) & mask

        bsz, _ = scores.shape
        final_indices = torch.full((bsz, k_max), -1, dtype=torch.long, device=scores.device)
        final_scores = torch.full((bsz, k_max), -1e9, dtype=torch.float32, device=scores.device)
        final_counts = torch.zeros((bsz,), dtype=torch.long, device=scores.device)

        for i in range(bsz):
            valid_scores = scores[i].clone()
            valid_scores[~mask[i]] = -1e9
            idx = torch.where(selected_mask[i])[0]
            if idx.numel() < min_select:
                n_valid = int(mask[i].sum().item())
                if n_valid > 0:
                    _, top_idx = torch.topk(valid_scores, k=min(min_select, n_valid))
                    idx = top_idx
            if idx.numel() > 0:
                sorted_idx = idx[torch.argsort(scores[i, idx], descending=True)]
                sorted_idx = sorted_idx[:k_max]
                n = sorted_idx.numel()
                final_indices[i, :n] = sorted_idx
                final_scores[i, :n] = scores[i, sorted_idx]
                final_counts[i] = n

        return {
            **out,
            "selected_candidate_positions": final_indices,
            "selected_scores": final_scores,
            "selected_counts": final_counts,
        }


def anes_stage1_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    margin: float = 0.5,
    lambda_neg: float = 1.0,
    lambda_rank: float = 0.2,
    lambda_len: float = 0.0,
    lambda_pseudo: float = 0.0,
    tau: float = 1.0,
) -> Dict[str, torch.Tensor]:

    scores = outputs["scores"]
    stop = outputs["stop_score"].unsqueeze(1)
    mask = batch["candidate_mask"]
    labels = batch.get("label", torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)).to(scores.device)
    weak = (batch["candidate_weak"] > 0.5) & mask
    nonweak = (~weak) & mask

    pos_margin = scores - stop
    weak_count = weak.float().sum(dim=1)
    pos_has_weak = (labels == 1) & (weak_count > 0)
    pos_no_weak = (labels == 1) & (weak_count <= 0) & mask.any(dim=1)
    neg_sample = (labels == 0) & mask.any(dim=1)

    if (weak & pos_has_weak.unsqueeze(1)).any():
        l_pos = F.softplus(margin - pos_margin[weak & pos_has_weak.unsqueeze(1)]).mean()
    else:
        l_pos = scores.sum() * 0.0


    neg_terms = []
    for i in torch.where(neg_sample)[0].tolist():
        s = scores[i][mask[i]]
        if s.numel() == 0:
            continue
        k = min(32, s.numel())
        top_s = torch.topk(s, k=k).values
        neg_terms.append(F.softplus(margin - (stop[i, 0] - top_s)).mean())
    l_neg = torch.stack(neg_terms).mean() if neg_terms else scores.sum() * 0.0

    rank_terms = []
    for i in torch.where(pos_has_weak)[0].tolist():
        p = scores[i][weak[i]]
        n = scores[i][nonweak[i]]
        if p.numel() == 0 or n.numel() == 0:
            continue
        if n.numel() > 128:
            top_n = torch.topk(n, k=128).values
        else:
            top_n = n
        rank_terms.append(F.softplus(-(p.view(-1, 1) - top_n.view(1, -1))).mean())
    l_rank = torch.stack(rank_terms).mean() if rank_terms else scores.sum() * 0.0


    pseudo_terms = []
    for i in torch.where(pos_no_weak)[0].tolist():
        s = scores[i].masked_fill(~mask[i], -1e9)
        top_idx = torch.argmax(s)
        pseudo_terms.append(F.softplus(margin - (scores[i, top_idx] - stop[i, 0])))
    l_pseudo = torch.stack(pseudo_terms).mean() if pseudo_terms else scores.sum() * 0.0

    probs = torch.sigmoid((scores - stop) / tau).masked_fill(~mask, 0.0)
    exp_k = probs.sum(dim=1)


    target_k = torch.zeros_like(exp_k)
    target_k = torch.where(pos_has_weak, weak_count.clamp(min=1.0, max=4.0), target_k)
    target_k = torch.where(pos_no_weak, torch.ones_like(target_k), target_k)

    l_len = (exp_k - target_k).pow(2).mean()

    loss = l_pos + lambda_neg * l_neg + lambda_rank * l_rank + lambda_len * l_len + lambda_pseudo * l_pseudo
    return {
        "loss": loss,
        "l_pos": l_pos.detach(),
        "l_neg": l_neg.detach(),
        "l_rank": l_rank.detach(),
        "l_len": l_len.detach(),
        "l_pseudo": l_pseudo.detach(),
    }
