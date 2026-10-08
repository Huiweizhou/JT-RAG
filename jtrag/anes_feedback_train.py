from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .anes_dataset import ANESTimeDataset, anes_collate, balanced_qid_sample
from .anes_model import ANESSelector, anes_stage1_loss
from .prompt_builder import build_prompt
from .llm_reward import QwenYesNoRunner


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _sample_masks_from_logits(
    logits: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int,
    k_max: int,
    temperature: float = 0.7,
) -> torch.Tensor:

    bsz, max_c = logits.shape
    if max_c == 0 or int(num_samples) <= 0:
        return torch.zeros((bsz, 0, max_c), dtype=torch.bool, device=logits.device)

    tau = max(1e-6, float(temperature))
    probs = torch.sigmoid(logits.detach() / tau).clamp(1e-6, 1.0 - 1e-6)
    rand = torch.rand((bsz, int(num_samples), max_c), dtype=probs.dtype, device=probs.device)
    sampled = (rand < probs.unsqueeze(1)) & mask.unsqueeze(1)

    if int(k_max) > 0:
        top_masks = torch.zeros_like(sampled)
        detached_logits = logits.detach()
        for b in range(bsz):
            for m in range(int(num_samples)):
                idx = torch.where(sampled[b, m])[0]
                if idx.numel() == 0:
                    continue
                sorted_idx = idx[torch.argsort(detached_logits[b, idx], descending=True)][: int(k_max)]
                top_masks[b, m, sorted_idx] = True
        sampled = top_masks
    return sampled


def _deterministic_mask_from_logits(logits: torch.Tensor, mask: torch.Tensor, k_max: int) -> torch.Tensor:

    selected = (logits.detach() > 0) & mask
    out = torch.zeros_like(selected)
    for b in range(logits.size(0)):
        idx = torch.where(selected[b])[0]
        if idx.numel() == 0:
            continue
        sorted_idx = idx[torch.argsort(logits.detach()[b, idx], descending=True)]
        if int(k_max) > 0:
            sorted_idx = sorted_idx[: int(k_max)]
        out[b, sorted_idx] = True
    return out


def _weak_mask(batch: Dict[str, torch.Tensor], k_max: int) -> torch.Tensor:

    weak = (batch["candidate_weak"] > 0.5) & batch["candidate_mask"]
    out = torch.zeros_like(weak)
    for b in range(weak.size(0)):
        idx = torch.where(weak[b])[0]
        if int(k_max) > 0:
            idx = idx[: int(k_max)]
        if idx.numel() > 0:
            out[b, idx] = True
    return out


def _log_pi_for_masks(probs: torch.Tensor, masks: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:

    eps = 1e-6
    p = probs.clamp(eps, 1.0 - eps)
    log_p = torch.log(p).unsqueeze(1)
    log_1mp = torch.log(1.0 - p).unsqueeze(1)
    valid = valid_mask.unsqueeze(1)
    selected = masks & valid
    return torch.where(selected, log_p, log_1mp).masked_fill(~valid, 0.0).sum(dim=-1)


def _redundancy_for_mask(candidate_sem: torch.Tensor, set_mask: torch.Tensor) -> float:

    idx = torch.where(set_mask)[0]
    if idx.numel() <= 1:
        return 0.0
    x = F.normalize(candidate_sem[idx].float(), dim=-1)
    sim = x @ x.t()
    n = int(idx.numel())
    value = (sim.sum() - sim.diag().sum()) / max(1, n * (n - 1))
    return float(value.detach().cpu())


def _mask_to_eids(batch: Dict[str, torch.Tensor], b: int, set_mask: torch.Tensor) -> List[int]:
    idx = torch.where(set_mask)[0].detach().cpu().tolist()
    cand_ids = batch["candidate_ids"][b].detach().cpu().tolist()
    return [int(cand_ids[p]) for p in idx if 0 <= p < len(cand_ids) and int(cand_ids[p]) >= 0]


def _build_reward_prompts(
    batch: Dict[str, torch.Tensor],
    set_masks: torch.Tensor,
    dataset: ANESTimeDataset,
) -> tuple[List[str], List[int], List[tuple[int, int]], List[int], List[float]]:

    sample_by_qid = {int(s.qid): s for s in dataset.samples}
    prompts: List[str] = []
    labels: List[int] = []
    positions: List[tuple[int, int]] = []
    lengths: List[int] = []
    redundancies: List[float] = []

    bsz, n_sets, _ = set_masks.shape
    for b in range(bsz):
        qid = int(batch["qid"][b].item())
        sample = sample_by_qid[qid]
        for s_idx in range(n_sets):
            sm = set_masks[b, s_idx]
            selected_eids = _mask_to_eids(batch, b, sm)
            prompts.append(
                build_prompt(
                    sample=sample,
                    selected_eids=selected_eids,
                    entities=dataset.entities,
                    graph=dataset.graph,
                )
            )
            labels.append(int(sample.label))
            positions.append((b, s_idx))
            lengths.append(len(selected_eids))
            redundancies.append(_redundancy_for_mask(batch["candidate_sem"][b].detach(), sm.detach()))
    return prompts, labels, positions, lengths, redundancies


def _preference_loss(log_pi: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
    if log_pi.numel() == 0 or log_pi.size(1) <= 1:
        return log_pi.sum() * 0.0
    best = torch.argmax(rewards, dim=1)
    worst = torch.argmin(rewards, dim=1)
    row = torch.arange(log_pi.size(0), device=log_pi.device)
    return F.softplus(-(log_pi[row, best] - log_pi[row, worst])).mean()


def _distill_loss(log_pi: torch.Tensor, rewards: torch.Tensor, reward_tau: float = 0.5, policy_tau: float = 1.0) -> torch.Tensor:
    if log_pi.numel() == 0 or log_pi.size(1) <= 1:
        return log_pi.sum() * 0.0
    teacher = torch.softmax(rewards.detach() / max(1e-6, float(reward_tau)), dim=1)
    student_logp = torch.log_softmax(log_pi / max(1e-6, float(policy_tau)), dim=1)
    return F.kl_div(student_logp, teacher, reduction="batchmean")


def _index_first_dim(obj: Dict[str, torch.Tensor], index: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:

    out: Dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == int(batch_size):
            out[key] = value[index]
        else:
            out[key] = value
    return out


def train_anes_with_llm_feedback_one_time(
    processed_root: Path,
    time_points: Sequence[int],
    sample_time: int,
    anes_model: ANESSelector,
    optimizer: torch.optim.Optimizer,
    qwen: QwenYesNoRunner,
    candidate_max_size: int = 256,
    train_pos_n: int = 1000,
    train_neg_n: int = 1000,
    seed: int = 2026,
    batch_size: int = 4,
    num_workers: int = 0,
    device: str = "cuda",
    k_max: int = 8,
    epochs: int = 1,
    num_sampled_sets: int = 4,
    include_deterministic_set: bool = True,
    include_weak_set: bool = True,
    gumbel_temperature: float = 0.7,
    score_batch_size: int = 4,
    beta_length: float = 0.05,
    gamma_redundancy: float = 0.0,
    lambda_anchor: float = 0.0,
    lambda_distill: float = 0.2,
    grad_clip: float = 1.0,
    show_progress: bool = True,
    max_invalid_score_ratio: float = 0.02,
    fail_on_invalid_score: bool = True,
    min_reward_spread: float = 1e-6,
) -> dict:

    qids = balanced_qid_sample(processed_root, sample_time, pos_n=train_pos_n, neg_n=train_neg_n, seed=seed + 17)
    dataset = ANESTimeDataset(
        processed_root=processed_root,
        time_points=time_points,
        sample_time=sample_time,
        candidate_max_size=candidate_max_size,
        qid_filter=qids,
        use_weak=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=anes_collate)

    anes_model.train().to(device)
    qwen.model.eval()
    for parameter in qwen.model.parameters():
        parameter.requires_grad_(False)

    history: List[dict] = []
    total_score_count_all = 0
    total_invalid_score_all = 0

    for epoch in range(1, int(epochs) + 1):
        pbar = tqdm(
            loader,
            desc=f"ANES-LLM反馈训练 | T={sample_time} | epoch={epoch}/{epochs}",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )
        total_loss = total_pref = total_distill = total_anchor = 0.0
        total_reward = total_reward_spread = 0.0
        total_sets = total_batches = total_valid_samples = total_informative_samples = 0

        for batch in pbar:
            batch = _move_batch_to_device(batch, device)
            out = anes_model(batch)
            logits = out["logits"]
            mask = batch["candidate_mask"]
            probs = out["probs"]

            mask_list = []
            sampled = _sample_masks_from_logits(
                logits,
                mask,
                num_sampled_sets,
                k_max=k_max,
                temperature=gumbel_temperature,
            )
            if sampled.size(1) > 0:
                mask_list.append(sampled)
            if include_deterministic_set:
                mask_list.append(_deterministic_mask_from_logits(logits, mask, k_max=k_max).unsqueeze(1))
            if include_weak_set:
                mask_list.append(_weak_mask(batch, k_max=k_max).unsqueeze(1))
            if not mask_list:
                continue

            set_masks = torch.cat(mask_list, dim=1)
            log_pi = _log_pi_for_masks(probs, set_masks, mask)
            prompts, labels, positions, lengths, redundancies = _build_reward_prompts(batch, set_masks, dataset)
            score_records = qwen.score_yes_no_batch(prompts, labels, score_batch_size=score_batch_size)
            if len(score_records) != len(positions):
                raise RuntimeError(
                    f"Qwen scorer 返回数量异常：expected={len(positions)}, actual={len(score_records)}。"
                )

            reward_tensor = torch.full_like(log_pi, float("nan"))
            score_valid = torch.zeros_like(log_pi, dtype=torch.bool)
            for rec, (b, s_idx), k_len, red in zip(score_records, positions, lengths, redundancies):
                valid = bool(int(rec.get("score_valid", 0)))
                logp = rec.get("log_prob_true_binary", None)
                valid = valid and logp is not None and math.isfinite(float(logp))
                if not valid:
                    continue
                red_value = float(red) if math.isfinite(float(red)) else 0.0


                reward = float(logp) - float(beta_length) * float(k_len) - float(gamma_redundancy) * red_value
                if not math.isfinite(reward):
                    continue
                reward_tensor[b, s_idx] = reward
                score_valid[b, s_idx] = True

            batch_score_count = int(score_valid.numel())
            batch_invalid_score = int((~score_valid).sum().item())
            total_score_count_all += batch_score_count
            total_invalid_score_all += batch_invalid_score
            invalid_ratio_so_far = total_invalid_score_all / max(1, total_score_count_all)
            if bool(fail_on_invalid_score) and invalid_ratio_so_far > float(max_invalid_score_ratio):
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    f"LLM feedback scorer 无效率过高：{invalid_ratio_so_far:.4%} > "
                    f"{float(max_invalid_score_ratio):.4%}。已中止，未保存该 feedback checkpoint。"
                )


            valid_sample = score_valid.all(dim=1)
            valid_index = torch.where(valid_sample)[0]
            if valid_index.numel() == 0:
                optimizer.zero_grad(set_to_none=True)
                continue

            bsz = int(log_pi.size(0))
            valid_log_pi = log_pi[valid_index]
            valid_reward = reward_tensor[valid_index]
            valid_out = _index_first_dim(out, valid_index, bsz)
            valid_batch = _index_first_dim(batch, valid_index, bsz)

            spread = valid_reward.max(dim=1).values - valid_reward.min(dim=1).values
            informative = spread > float(min_reward_spread)
            if informative.any():
                l_pref = _preference_loss(valid_log_pi[informative], valid_reward[informative])
                l_distill = (
                    _distill_loss(valid_log_pi[informative], valid_reward[informative])
                    if float(lambda_distill) > 0
                    else valid_log_pi.sum() * 0.0
                )
            else:
                l_pref = valid_log_pi.sum() * 0.0
                l_distill = valid_log_pi.sum() * 0.0


            l_anchor = (
                anes_stage1_loss(valid_out, valid_batch)["loss"]
                if float(lambda_anchor) > 0 else valid_log_pi.sum() * 0.0
            )
            loss = l_pref + float(lambda_distill) * l_distill + float(lambda_anchor) * l_anchor
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(anes_model.parameters(), float(grad_clip))
            optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_pref += float(l_pref.detach().cpu())
            total_distill += float(l_distill.detach().cpu())
            total_anchor += float(l_anchor.detach().cpu())
            total_reward += float(valid_reward.mean().detach().cpu())
            total_reward_spread += float(spread.mean().detach().cpu())
            total_sets += int(valid_reward.numel())
            total_valid_samples += int(valid_index.numel())
            total_informative_samples += int(informative.sum().item())
            total_batches += 1

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / max(1, total_batches):.4f}",
                    "pref": f"{total_pref / max(1, total_batches):.4f}",
                    "rew": f"{total_reward / max(1, total_batches):.4f}",
                    "score_bad": f"{invalid_ratio_so_far:.2%}",
                }
            )

        metrics = {
            "sample_time": int(sample_time),
            "context_time": int(dataset.context_time),
            "epoch": int(epoch),
            "loss": total_loss / max(1, total_batches),
            "l_pref": total_pref / max(1, total_batches),
            "l_distill": total_distill / max(1, total_batches),
            "l_anchor": total_anchor / max(1, total_batches),
            "reward_mean": total_reward / max(1, total_batches),
            "reward_spread_mean": total_reward_spread / max(1, total_batches),
            "num_reward_sets": int(total_sets),
            "num_valid_reward_samples": int(total_valid_samples),
            "num_informative_reward_samples": int(total_informative_samples),
            "scorer_invalid_ratio": total_invalid_score_all / max(1, total_score_count_all),
            "num_samples": len(dataset),
        }
        history.append(metrics)
        print(
            f"[ANES-LLM反馈][T={sample_time}][epoch {epoch}/{epochs}] "
            f"loss={metrics['loss']:.4f} pref={metrics['l_pref']:.4f} "
            f"distill={metrics['l_distill']:.4f} anchor={metrics['l_anchor']:.4f} "
            f"reward={metrics['reward_mean']:.4f} spread={metrics['reward_spread_mean']:.4f} "
            f"scorer_invalid={metrics['scorer_invalid_ratio']:.2%}"
        )

    return {"time": int(sample_time), "context_time": int(dataset.context_time), "history": history}
