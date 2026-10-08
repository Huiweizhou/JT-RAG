from __future__ import annotations

import gc
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .anes_dataset import ANESTimeDataset, anes_collate
from .anes_feedback_train import train_anes_with_llm_feedback_one_time
from .anes_io import load_semantic_embeddings, load_struct_embeddings, previous_time, write_json
from .anes_model import ANESConfig, ANESSelector, anes_stage1_loss
from .llm_reward import QwenYesNoRunner
from .validation import check_validation_inputs, evaluate_next_validation, record_validation_summary, validation_target
from .resume_utils import (
    find_latest_epoch_checkpoint,
    get_stage_meta,
    init_state,
    is_completed,
    is_valid_lora_adapter,
    load_anes_checkpoint,
    mark_completed,
    mark_running,
    read_json_safe,
    maybe_load_completed_stage_ckpt,
    remove_incomplete_lora_adapter,
    save_anes_checkpoint,
    save_state,
    stage_key,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:

    value = float(lr)
    for group in optimizer.param_groups:
        group["lr"] = value


def infer_dims(processed_root: Path, time_points: Sequence[int], first_sample_time: int) -> tuple[int, int]:

    sem = load_semantic_embeddings(processed_root, mmap_mode="r")
    ctx = previous_time([int(t) for t in time_points], int(first_sample_time))
    struct = load_struct_embeddings(processed_root, ctx, mmap_mode="r")
    return int(sem.shape[1]), int(struct.shape[1])


def build_anes_model(cfg: dict) -> ANESSelector:
    sem_dim, str_dim = infer_dims(Path(cfg["PROCESSED_ROOT"]), cfg["TIME_POINTS"], cfg["INIT_TRAIN_TIMES"][0])
    model_cfg = ANESConfig(
        sem_dim=sem_dim,
        str_dim=str_dim,
        tmp_dim=10,
        meta_dim=13,
        hidden_dim=int(cfg["HIDDEN_DIM"]),
        num_heads=int(cfg["NUM_HEADS"]),
        set_attn_layers=int(cfg["SET_ATTN_LAYERS"]),
        dropout=float(cfg["DROPOUT"]),
        use_pair_query_tokens=bool(cfg.get("USE_PAIR_QUERY_TOKENS", False)),
        use_candidate_query_cross_attn=bool(cfg.get("USE_CANDIDATE_QUERY_CROSS_ATTN", True)),
        use_side_embedding=bool(cfg.get("USE_SIDE_EMBEDDING", True)),
        use_enhanced_stop=bool(cfg.get("USE_ENHANCED_STOP", True)),
        candidate_query_attn_chunk_size=int(cfg.get("CANDIDATE_QUERY_ATTN_CHUNK_SIZE", 2048)),
    )
    return ANESSelector(model_cfg)


def save_checkpoint(
    path: Path,
    model: ANESSelector,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    sample_time: int,
    stage: str,
    epoch: Optional[int] = None,
    stage_key_value: Optional[str] = None,
    completed: bool = False,
) -> None:

    save_anes_checkpoint(
        path=path,
        model=model,
        optimizer=optimizer,
        cfg=cfg,
        sample_time=sample_time,
        stage=stage,
        epoch=epoch,
        stage_key_value=stage_key_value,
        completed=completed,
    )


def make_dataloader(dataset, cfg: dict, shuffle: bool, batch_size: Optional[int] = None) -> DataLoader:

    num_workers = int(cfg.get("ANES_NUM_WORKERS", 0))
    pin_memory = bool(cfg.get("PIN_MEMORY", False)) and str(cfg.get("DEVICE", "cpu")).startswith("cuda")
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size or cfg.get("ANES_BATCH_SIZE", 16)),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": anes_collate,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(cfg.get("PERSISTENT_WORKERS", True))
        kwargs["prefetch_factor"] = int(cfg.get("PREFETCH_FACTOR", 2))
    return DataLoader(**kwargs)


def get_amp_dtype(cfg: dict) -> torch.dtype:
    if str(cfg.get("ANES_AMP_DTYPE", "float16")).lower() == "bfloat16":
        return torch.bfloat16
    return torch.float16


def build_qwen_runner(cfg: dict, adapter_path: Optional[Path] = None) -> QwenYesNoRunner:

    print(f"加载 Qwen：{cfg['QWEN_MODEL_PATH']}")
    if adapter_path is not None:
        print(f"加载 LoRA adapter：{adapter_path}")
    return QwenYesNoRunner(
        model_name_or_path=str(cfg["QWEN_MODEL_PATH"]),
        device=str(cfg["DEVICE"]),
        torch_dtype=str(cfg["QWEN_TORCH_DTYPE"]),
        adapter_path=None if adapter_path is None else str(adapter_path),
        context_limit=int(cfg["LLM_CONTEXT_LIMIT"]),
    )


def unload_qwen_runner(qwen_runner: Optional[QwenYesNoRunner]) -> None:

    if qwen_runner is not None:
        try:
            del qwen_runner.model
            del qwen_runner.tokenizer
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_anes_one_time(
    model: ANESSelector,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    sample_time: int,
    epochs: int,
    stage: str,
    start_epoch: int = 0,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_prefix: Optional[str] = None,
    stage_key_value: Optional[str] = None,
) -> dict:

    dataset = ANESTimeDataset(
        processed_root=Path(cfg["PROCESSED_ROOT"]),
        time_points=cfg["TIME_POINTS"],
        sample_time=int(sample_time),
        candidate_max_size=int(cfg["CANDIDATE_MAX_SIZE"]),
        use_weak=True,
    )
    loader = make_dataloader(dataset, cfg, shuffle=True)
    device = str(cfg["DEVICE"])
    model.train().to(device)
    amp_enabled = bool(cfg.get("ANES_AMP", False)) and device.startswith("cuda")
    amp_dtype = get_amp_dtype(cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)

    start_epoch = int(start_epoch or 0)
    total_epochs = int(epochs)
    if start_epoch >= total_epochs:
        print(f"[ANES][{stage}][T={sample_time}] 已完成 {start_epoch}/{total_epochs} 个 epoch，跳过训练。")
        return {"time": int(sample_time), "context_time": int(dataset.context_time), "history": [], "skipped": True}

    history: List[dict] = []
    for epoch in range(start_epoch + 1, total_epochs + 1):
        total_loss = 0.0
        total_pos = 0.0
        total_neg = 0.0
        total_rank = 0.0
        total_len = 0.0
        total_pseudo = 0.0
        total_batches = 0
        total_candidates = 0
        total_weak = 0
        total_weak_possible = 0
        total_weak_retained = 0

        pbar = tqdm(
            loader,
            desc=f"ANES训练 | {stage} | T={sample_time} | epoch={epoch}/{total_epochs}",
            dynamic_ncols=True,
            leave=True,
            disable=not bool(cfg.get("SHOW_PROGRESS", True)),
        )

        for batch in pbar:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                out = model(batch)
                loss_dict = anes_stage1_loss(
                    out,
                    batch,
                    margin=float(cfg["MARGIN"]),
                    lambda_neg=float(cfg["LAMBDA_NEG"]),
                    lambda_rank=float(cfg["LAMBDA_RANK"]),
                    lambda_len=float(cfg["LAMBDA_LEN"]),
                    lambda_pseudo=float(cfg.get("LAMBDA_PSEUDO", 0.0)),
                )
                loss = loss_dict["loss"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["GRAD_CLIP"]))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["GRAD_CLIP"]))
                optimizer.step()

            total_loss += float(loss.detach().cpu())
            total_pos += float(loss_dict["l_pos"].cpu())
            total_neg += float(loss_dict["l_neg"].cpu())
            total_rank += float(loss_dict["l_rank"].cpu())
            total_len += float(loss_dict["l_len"].cpu())
            total_pseudo += float(loss_dict.get("l_pseudo", torch.tensor(0.0)).cpu())
            total_batches += 1
            batch_candidates = int(batch["candidate_mask"].sum().detach().cpu())
            batch_weak = int(batch["candidate_weak"].sum().detach().cpu())
            total_candidates += batch_candidates
            total_weak += batch_weak
            total_weak_possible += int(batch.get("weak_total_in_full_candidate", torch.zeros(1, device=device)).sum().detach().cpu())
            total_weak_retained += int(batch.get("weak_retained_count", torch.zeros(1, device=device)).sum().detach().cpu())

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / max(1, total_batches):.4f}",
                    "pos": f"{total_pos / max(1, total_batches):.4f}",
                    "neg": f"{total_neg / max(1, total_batches):.4f}",
                    "rank": f"{total_rank / max(1, total_batches):.4f}",
                    "len": f"{total_len / max(1, total_batches):.4f}",
                    "pseudo": f"{total_pseudo / max(1, total_batches):.4f}",
                    "weak": total_weak,
                }
            )

        metrics = {
            "stage": stage,
            "sample_time": int(sample_time),
            "context_time": int(dataset.context_time),
            "epoch": int(epoch),
            "loss": total_loss / max(1, total_batches),
            "l_pos": total_pos / max(1, total_batches),
            "l_neg": total_neg / max(1, total_batches),
            "l_rank": total_rank / max(1, total_batches),
            "l_len": total_len / max(1, total_batches),
            "l_pseudo": total_pseudo / max(1, total_batches),
            "avg_candidates_per_batch": total_candidates / max(1, total_batches),
            "weak_labels_seen": total_weak,
            "weak_total_in_full_candidate": int(total_weak_possible),
            "weak_retained_after_prefilter": int(total_weak_retained),
            "weak_candidate_coverage": float(total_weak_retained / max(1, total_weak_possible)),
            "num_samples": len(dataset),
        }
        history.append(metrics)
        print(
            f"[ANES][{stage}][T={sample_time}][epoch {epoch}/{total_epochs}] "
            f"loss={metrics['loss']:.4f} pos={metrics['l_pos']:.4f} neg={metrics['l_neg']:.4f} "
            f"rank={metrics['l_rank']:.4f} len={metrics['l_len']:.4f} pseudo={metrics['l_pseudo']:.4f} weak={total_weak}"
        )

        if bool(cfg.get("SAVE_ANES_EVERY_EPOCH", True)) and checkpoint_dir is not None and checkpoint_prefix:
            epoch_ckpt = Path(checkpoint_dir) / f"{checkpoint_prefix}_epoch{int(epoch)}.pt"
            save_checkpoint(
                epoch_ckpt,
                model,
                optimizer,
                cfg,
                int(sample_time),
                stage=stage,
                epoch=int(epoch),
                stage_key_value=stage_key_value,
                completed=False,
            )
            print(f"[ANES][{stage}][T={sample_time}] 已保存 epoch checkpoint：{epoch_ckpt}")

    return {"time": int(sample_time), "context_time": int(dataset.context_time), "history": history}


def main(cfg: dict) -> int:
    cfg = dict(cfg)
    if cfg["DEVICE"] == "auto":
        cfg["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"


    init_times = [int(t) for t in cfg.get("INIT_TRAIN_TIMES", [])]
    joint_times = [int(t) for t in cfg.get("JOINT_TRAIN_TIMES", [])]
    overlap = sorted(set(init_times) & set(joint_times))
    if overlap:
        raise ValueError(f"INIT_TRAIN_TIMES 与 JOINT_TRAIN_TIMES 存在重叠时间步：{overlap}。请保证前五个时间步只训练 ANES。")
    if init_times and joint_times and min(joint_times) <= max(init_times):
        raise ValueError(
            f"JOINT_TRAIN_TIMES 应晚于 INIT_TRAIN_TIMES。当前 max(INIT)={max(init_times)}, min(JOINT)={min(joint_times)}。"
        )
    if bool(cfg.get("USE_PAIR_QUERY_TOKENS", False)):
        raise ValueError("当前方法要求仅使用原生向量，USE_PAIR_QUERY_TOKENS 必须为 False。")

    if int(cfg.get("EVIDENCE_K_MAX", 0)) <= 0:
        raise ValueError("EVIDENCE_K_MAX 必须大于 0，用于控制证据数量和完整 Prompt 长度。")
    if int(cfg.get("LORA_CONTEXT_LIMIT", 0)) <= 0 or int(cfg.get("LLM_CONTEXT_LIMIT", 0)) <= 0:
        raise ValueError("LORA_CONTEXT_LIMIT 和 LLM_CONTEXT_LIMIT 必须为正整数。")

    set_seed(int(cfg["SEED"]))

    check_validation_inputs(cfg)

    if bool(cfg.get("ALLOW_TF32", True)) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    out_dir = Path(cfg["OUTPUT_DIR"])
    ckpt_dir = out_dir / "checkpoints"
    lora_dir = out_dir / "qwen_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lora_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", cfg)

    resume_enabled = bool(cfg.get("RESUME", True))
    state_path = out_dir / "stage_state.json"
    state = init_state(state_path, cfg) if resume_enabled else {
        "version": 1,
        "last_anes_ckpt": None,
        "active_lora_adapter": None,
        "completed_stages": {},
        "running_stage": None,
    }

    train_log_path = out_dir / "train_log.json"
    old_log_obj = read_json_safe(train_log_path, default={}) if resume_enabled else {}
    all_logs: List[dict] = list(old_log_obj.get("logs", [])) if isinstance(old_log_obj.get("logs", []), list) else []

    print("=" * 80)
    print("开始 JT-RAG / ANES-Qwen v4 训练")
    print(f"数据目录：{cfg['PROCESSED_ROOT']}")
    print(f"输出目录：{out_dir}")
    print(f"设备：{cfg['DEVICE']}")
    print(f"前五个时间步只训练 ANES：{cfg['INIT_TRAIN_TIMES']}")
    print(f"后续时间步 ANES-LLM 交替训练：{cfg['JOINT_TRAIN_TIMES']}")
    print(f"断点续训：{'开启' if resume_enabled else '关闭'}")
    print("=" * 80)

    model = build_anes_model(cfg)
    model.to(cfg["DEVICE"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["ANES_INIT_LR"]),
        weight_decay=float(cfg["WEIGHT_DECAY"]),
    )

    last_ckpt: Optional[Path] = None
    active_lora_adapter: Optional[Path] = None

    if resume_enabled:
        last_ckpt_str = state.get("last_anes_ckpt")
        if last_ckpt_str and Path(last_ckpt_str).exists():
            load_anes_checkpoint(Path(last_ckpt_str), model, optimizer, device=str(cfg["DEVICE"]))
            last_ckpt = Path(last_ckpt_str)
        adapter_str = state.get("active_lora_adapter")
        if adapter_str and is_valid_lora_adapter(Path(adapter_str)):
            active_lora_adapter = Path(adapter_str)
            print(f"[Resume] 当前活动 LoRA adapter：{active_lora_adapter}")

    def _write_train_log() -> None:
        write_json(train_log_path, {"logs": all_logs})

    def _append_log(entry: dict) -> None:
        all_logs.append(entry)
        _write_train_log()


    def _load_latest_epoch_if_any(prefix: str, stage_total_epochs: int) -> int:

        if not resume_enabled:
            return 0
        epoch_ckpt, start_epoch = find_latest_epoch_checkpoint(ckpt_dir, prefix)
        if epoch_ckpt is not None and start_epoch > 0 and start_epoch < int(stage_total_epochs):
            load_anes_checkpoint(epoch_ckpt, model, optimizer, device=str(cfg["DEVICE"]))
            print(f"[Resume] 阶段 {prefix} 将从 epoch {start_epoch + 1}/{stage_total_epochs} 继续。")
            return int(start_epoch)
        if epoch_ckpt is not None and start_epoch >= int(stage_total_epochs):
            load_anes_checkpoint(epoch_ckpt, model, optimizer, device=str(cfg["DEVICE"]))
            print(f"[Resume] 阶段 {prefix} 已有完整 epoch checkpoint，将直接保存最终阶段 checkpoint。")
            return int(start_epoch)
        return 0


    for t in cfg["INIT_TRAIN_TIMES"]:
        t = int(t)
        set_optimizer_lr(optimizer, float(cfg["ANES_INIT_LR"]))
        key = stage_key("init_anes_only", t)
        final_ckpt = ckpt_dir / f"anes_{t}.pt"
        print("\n" + "-" * 80)
        print(f"阶段 1：只训练 ANES，样本时间 T={t}")

        if resume_enabled and is_completed(state, key):
            print(f"[Resume] 跳过已完成阶段：{key}")
            maybe_load_completed_stage_ckpt(
                state,
                key,
                model,
                optimizer,
                device=str(cfg["DEVICE"]),
                load_model=bool(cfg.get("RESUME_LOAD_COMPLETED_STAGE_CKPT", True)),
            )
            last_ckpt = Path(get_stage_meta(state, key).get("ckpt_path", final_ckpt))
            continue

        if resume_enabled:
            state = mark_running(state_path, state, key, {"sample_time": t, "stage": "init_anes_only"})

        prefix = f"anes_{t}_init_anes_only"
        start_epoch = _load_latest_epoch_if_any(prefix, int(cfg["INIT_EPOCHS_PER_TIME"]))
        train_log = train_anes_one_time(
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            sample_time=t,
            epochs=int(cfg["INIT_EPOCHS_PER_TIME"]),
            stage="init_anes_only",
            start_epoch=start_epoch,
            checkpoint_dir=ckpt_dir,
            checkpoint_prefix=prefix,
            stage_key_value=key,
        )
        save_checkpoint(final_ckpt, model, optimizer, cfg, t, stage="init_anes_only", stage_key_value=key, completed=True)
        last_ckpt = final_ckpt
        print(f"已保存 ANES checkpoint：{final_ckpt}")
        _append_log({"time": t, "stage": "init_anes_only", "train": train_log, "ckpt": str(final_ckpt)})
        if resume_enabled:
            state = mark_completed(state_path, state, key, {"ckpt_path": str(final_ckpt), "sample_time": t})


    for t in cfg["JOINT_TRAIN_TIMES"]:
        t = int(t)
        print("\n" + "-" * 80)
        print(f"阶段 2：ANES-LLM 交替训练时间步 T={t}")


        set_optimizer_lr(optimizer, float(cfg["ANES_WEAK_LR"]))
        weak_key = stage_key("joint_anes_weak_refresh", t)
        weak_ckpt = ckpt_dir / f"anes_{t}_weak.pt"
        weak_train_log = None
        if resume_enabled and is_completed(state, weak_key):
            print(f"[Resume] 跳过已完成阶段：{weak_key}")
            maybe_load_completed_stage_ckpt(
                state,
                weak_key,
                model,
                optimizer,
                device=str(cfg["DEVICE"]),
                load_model=bool(cfg.get("RESUME_LOAD_COMPLETED_STAGE_CKPT", True)),
            )
        else:
            if resume_enabled:
                state = mark_running(state_path, state, weak_key, {"sample_time": t, "stage": "joint_anes_weak_refresh"})
            prefix = f"anes_{t}_joint_anes_weak_refresh"
            start_epoch = _load_latest_epoch_if_any(prefix, int(cfg["JOINT_ANES_WEAK_EPOCHS_PER_TIME"]))
            weak_train_log = train_anes_one_time(
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                sample_time=t,
                epochs=int(cfg["JOINT_ANES_WEAK_EPOCHS_PER_TIME"]),
                stage="joint_anes_weak_refresh",
                start_epoch=start_epoch,
                checkpoint_dir=ckpt_dir,
                checkpoint_prefix=prefix,
                stage_key_value=weak_key,
            )
            save_checkpoint(weak_ckpt, model, optimizer, cfg, t, stage="joint_anes_weak_refresh", stage_key_value=weak_key, completed=True)
            last_ckpt = weak_ckpt
            print(f"已保存 weak refresh 后 ANES checkpoint：{weak_ckpt}")
            if resume_enabled:
                state = mark_completed(state_path, state, weak_key, {"ckpt_path": str(weak_ckpt), "sample_time": t})

        joint_round_logs: List[dict] = []
        qwen_runner: Optional[QwenYesNoRunner] = None

        for round_idx in range(1, int(cfg["JOINT_ROUNDS_PER_TIME"]) + 1):
            print("\n" + "." * 80)
            print(f"时间步 T={t} | 交替轮次 {round_idx}/{cfg['JOINT_ROUNDS_PER_TIME']}")


            lora_stats = None
            lora_key = stage_key("qwen_lora", t, round_idx)
            adapter_out = lora_dir / f"adapter_{t}_round{round_idx}"

            if bool(cfg.get("ENABLE_QWEN_LORA_IN_JOINT_STAGE", True)):
                if resume_enabled and bool(cfg.get("REMOVE_INCOMPLETE_LORA_ON_RESUME", True)):
                    remove_incomplete_lora_adapter(adapter_out)

                if resume_enabled and is_completed(state, lora_key):
                    meta = get_stage_meta(state, lora_key)
                    adapter_path = Path(meta.get("adapter_path", adapter_out))
                    if is_valid_lora_adapter(adapter_path):
                        active_lora_adapter = adapter_path
                        lora_stats = {"adapter_path": str(adapter_path), "skipped_by_resume": True}
                        print(f"[Resume] 跳过已完成 LoRA 阶段：{lora_key} | adapter={adapter_path}")
                    else:
                        print(f"[Resume] LoRA 阶段记录存在但 adapter 不完整，将重训：{adapter_path}")
                        state.get("completed_stages", {}).pop(lora_key, None)
                        save_state(state_path, state)

                if lora_stats is None:
                    unload_qwen_runner(qwen_runner)
                    qwen_runner = None
                    from .qwen_lora_train import train_qwen_lora_on_time

                    if resume_enabled:
                        state = mark_running(state_path, state, lora_key, {"sample_time": t, "round": round_idx})
                    print(f"开始 Qwen LoRA 适配：T={t}, round={round_idx}, 输出={adapter_out}")
                    adapter_path = train_qwen_lora_on_time(
                        processed_root=Path(cfg["PROCESSED_ROOT"]),
                        time_points=cfg["TIME_POINTS"],
                        sample_time=t,
                        anes_model=model,
                        qwen_model_path=str(cfg["QWEN_MODEL_PATH"]),
                        output_adapter_dir=adapter_out,
                        previous_adapter_dir=active_lora_adapter,
                        candidate_max_size=int(cfg["CANDIDATE_MAX_SIZE"]),
                        train_pos_n=int(cfg["LORA_TRAIN_POS_N"]),
                        train_neg_n=int(cfg["LORA_TRAIN_NEG_N"]),
                        seed=int(cfg["SEED"]) + round_idx,
                        k_max=int(cfg["EVIDENCE_K_MAX"]),
                        context_limit=int(cfg["LORA_CONTEXT_LIMIT"]),
                        epochs=int(cfg["LORA_EPOCHS"]),
                        batch_size=int(cfg["LORA_BATCH_SIZE"]),
                        grad_accum=int(cfg["LORA_GRAD_ACCUM"]),
                        lr=float(cfg["LORA_LR"]),
                        device=str(cfg["DEVICE"]),
                        show_progress=bool(cfg["SHOW_PROGRESS"]),
                        torch_dtype=str(cfg.get("LORA_TORCH_DTYPE", "auto")),
                        gradient_checkpointing=bool(cfg.get("LORA_GRADIENT_CHECKPOINTING", False)),
                        fail_on_high_nan_ratio=bool(cfg.get("LORA_FAIL_ON_HIGH_NAN_RATIO", True)),
                        max_nan_ratio=float(cfg.get("LORA_MAX_NAN_RATIO", 0.05)),
                    )
                    active_lora_adapter = Path(adapter_path)
                    lora_stats = {"adapter_path": str(adapter_path)}
                    print(f"Qwen LoRA adapter 已保存：{adapter_path}")
                    if resume_enabled:
                        state = mark_completed(
                            state_path,
                            state,
                            lora_key,
                            {"adapter_path": str(adapter_path), "sample_time": t, "round": round_idx},
                        )


            unload_qwen_runner(qwen_runner)
            qwen_runner = build_qwen_runner(cfg, adapter_path=active_lora_adapter)


            feedback_log = None
            feedback_key = stage_key("anes_llm_feedback", t, round_idx)
            feedback_ckpt = ckpt_dir / f"anes_{t}_round{round_idx}_feedback.pt"
            if bool(cfg.get("ENABLE_LLM_FEEDBACK", True)):
                if resume_enabled and is_completed(state, feedback_key):
                    print(f"[Resume] 跳过已完成反馈阶段：{feedback_key}")
                    maybe_load_completed_stage_ckpt(
                        state,
                        feedback_key,
                        model,
                        optimizer,
                        device=str(cfg["DEVICE"]),
                        load_model=bool(cfg.get("RESUME_LOAD_COMPLETED_STAGE_CKPT", True)),
                    )
                    feedback_log = {"skipped_by_resume": True}
                else:
                    if resume_enabled:
                        state = mark_running(state_path, state, feedback_key, {"sample_time": t, "round": round_idx})
                    set_optimizer_lr(optimizer, float(cfg["ANES_FEEDBACK_LR"]))
                    feedback_log = train_anes_with_llm_feedback_one_time(
                        processed_root=Path(cfg["PROCESSED_ROOT"]),
                        time_points=cfg["TIME_POINTS"],
                        sample_time=t,
                        anes_model=model,
                        optimizer=optimizer,
                        qwen=qwen_runner,
                        candidate_max_size=int(cfg["CANDIDATE_MAX_SIZE"]),
                        train_pos_n=int(cfg["FEEDBACK_TRAIN_POS_N"]),
                        train_neg_n=int(cfg["FEEDBACK_TRAIN_NEG_N"]),
                        seed=int(cfg["SEED"]) + round_idx * 100,
                        batch_size=int(cfg["FEEDBACK_BATCH_SIZE"]),
                        num_workers=int(cfg["FEEDBACK_NUM_WORKERS"]),
                        device=str(cfg["DEVICE"]),
                        k_max=int(cfg["EVIDENCE_K_MAX"]),
                        epochs=int(cfg["FEEDBACK_EPOCHS"]),
                        num_sampled_sets=int(cfg["FEEDBACK_NUM_SAMPLED_SETS"]),
                        include_deterministic_set=bool(cfg["FEEDBACK_INCLUDE_DETERMINISTIC_SET"]),
                        include_weak_set=bool(cfg["FEEDBACK_INCLUDE_WEAK_SET"]),
                        gumbel_temperature=float(cfg["FEEDBACK_GUMBEL_TEMPERATURE"]),
                        score_batch_size=int(cfg["FEEDBACK_SCORE_BATCH_SIZE"]),
                        beta_length=float(cfg["FEEDBACK_BETA_LENGTH"]),
                        gamma_redundancy=float(cfg["FEEDBACK_GAMMA_REDUNDANCY"]),
                        lambda_anchor=float(cfg["FEEDBACK_LAMBDA_ANCHOR"]),
                        lambda_distill=float(cfg["FEEDBACK_LAMBDA_DISTILL"]),
                        grad_clip=float(cfg["GRAD_CLIP"]),
                        show_progress=bool(cfg["SHOW_PROGRESS"]),
                        max_invalid_score_ratio=float(cfg.get("FEEDBACK_MAX_INVALID_SCORE_RATIO", 0.02)),
                        fail_on_invalid_score=bool(cfg.get("FEEDBACK_FAIL_ON_INVALID_SCORE", True)),
                        min_reward_spread=float(cfg.get("FEEDBACK_MIN_REWARD_SPREAD", 1e-6)),
                    )
                    save_checkpoint(feedback_ckpt, model, optimizer, cfg, t, stage="joint_llm_feedback", stage_key_value=feedback_key, completed=True)
                    last_ckpt = feedback_ckpt
                    print(f"已保存 LLM feedback 后 ANES checkpoint：{feedback_ckpt}")
                    if resume_enabled:
                        state = mark_completed(
                            state_path,
                            state,
                            feedback_key,
                            {"ckpt_path": str(feedback_ckpt), "sample_time": t, "round": round_idx},
                        )

            joint_round_logs.append({"round": int(round_idx), "lora": lora_stats, "feedback": feedback_log})
            _write_train_log()


        final_key = stage_key("joint_final", t)
        final_ckpt = ckpt_dir / f"anes_{t}_final.pt"
        if resume_enabled and is_completed(state, final_key):
            print(f"[Resume] 跳过已完成最终保存阶段：{final_key}")
            maybe_load_completed_stage_ckpt(
                state,
                final_key,
                model,
                optimizer,
                device=str(cfg["DEVICE"]),
                load_model=bool(cfg.get("RESUME_LOAD_COMPLETED_STAGE_CKPT", True)),
            )
            last_ckpt = Path(get_stage_meta(state, final_key).get("ckpt_path", final_ckpt))
        else:
            if resume_enabled:
                state = mark_running(state_path, state, final_key, {"sample_time": t})
            save_checkpoint(final_ckpt, model, optimizer, cfg, t, stage="joint_final", stage_key_value=final_key, completed=True)
            last_ckpt = final_ckpt
            if resume_enabled:
                state = mark_completed(state_path, state, final_key, {"ckpt_path": str(final_ckpt), "sample_time": t})
            _append_log(
                {
                    "time": t,
                    "stage": "joint_anes_llm",
                    "weak_refresh": weak_train_log,
                    "rounds": joint_round_logs,
                    "final_ckpt": str(final_ckpt),
                    "active_lora_adapter": None if active_lora_adapter is None else str(active_lora_adapter),
                }
            )

        valid_time, _ = validation_target(cfg["PROCESSED_ROOT"], cfg["TIME_POINTS"], t)
        valid_dir = out_dir / "validation"
        valid_csv = valid_dir / f"train_{t}_valid_{valid_time}.csv"
        valid_key = stage_key("next_time_validation", t)
        valid_metrics = None
        if resume_enabled and is_completed(state, valid_key) and valid_csv.is_file() and valid_csv.with_suffix(".metrics.json").is_file():
            valid_metrics = get_stage_meta(state, valid_key).get("metrics")
        if valid_metrics is None:
            if qwen_runner is None:
                qwen_runner = build_qwen_runner(cfg, adapter_path=active_lora_adapter)
            if resume_enabled:
                state = mark_running(state_path, state, valid_key, {"train_time": t, "valid_time": valid_time})
            valid_metrics = evaluate_next_validation(model, qwen_runner, cfg, t, valid_csv)
            if resume_enabled:
                state = mark_completed(state_path, state, valid_key, {"metrics": valid_metrics})
            _append_log({"time": t, "stage": "next_time_validation", "metrics": valid_metrics})
        record_validation_summary(valid_dir, valid_metrics)
        print(f"[Validation {t} -> {valid_time}] "
              f"Accuracy={valid_metrics['accuracy']:.4f} Precision={valid_metrics['precision']:.4f} "
              f"Recall={valid_metrics['recall']:.4f} F1={valid_metrics['f1']:.4f}")
        unload_qwen_runner(qwen_runner)

    print("\n" + "=" * 80)
    print("JT-RAG / ANES-Qwen v4 训练完成。")
    print(f"最终 ANES checkpoint：{last_ckpt}")
    print(f"最终 Qwen LoRA adapter：{active_lora_adapter}")
    print(f"日志：{train_log_path}")
    print(f"断点状态：{state_path}")
    print("=" * 80)
    return 0
