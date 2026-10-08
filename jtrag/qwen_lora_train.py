from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .anes_dataset import ANESTimeDataset, anes_collate, balanced_qid_sample
from .prompt_builder import build_prompt, format_chat_prompt


def _resolve_qwen_dtype(torch_dtype: str, device: str):

    name = str(torch_dtype or "auto").lower()
    if device != "cuda":
        return torch.float32, "float32"
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16, "bfloat16"
    if name in {"fp32", "float32"}:
        return torch.float32, "float32"
    return torch.float16, "float16"


def _supervised_token_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:

    labels = batch["labels"]
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask", None),
        use_cache=False,
    )
    logits = outputs.logits
    if logits.size(1) < 2:
        raise RuntimeError("sequence too short for causal LM loss")
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    if int(mask.sum().detach().cpu()) == 0:
        raise ValueError("empty supervised labels")
    selected_logits = shift_logits[mask]
    selected_labels = shift_labels[mask]
    if not torch.isfinite(selected_logits).all():
        raise FloatingPointError("non-finite logits on supervised positions")
    loss = F.cross_entropy(selected_logits.float(), selected_labels, reduction="mean")
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite supervised loss")
    return loss


class PromptAnswerDataset(Dataset):


    def __init__(self, examples: List[dict], tokenizer, context_limit: int = 8192) -> None:
        self.examples = examples
        self.tokenizer = tokenizer
        self.context_limit = int(context_limit)
        if self.context_limit <= 0:
            raise ValueError("context_limit 必须为正整数。")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        prompt_text = ex["prompt_text"]
        answer_text = ex["answer_text"]

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False, truncation=False)["input_ids"]
        answer_ids = self.tokenizer(answer_text, add_special_tokens=False, truncation=False)["input_ids"]
        if len(answer_ids) == 0:
            answer_ids = self.tokenizer(
                " " + answer_text.strip(), add_special_tokens=False, truncation=False
            )["input_ids"]
        if len(answer_ids) == 0:
            answer_ids = [self.tokenizer.eos_token_id or self.tokenizer.pad_token_id]

        input_ids = list(prompt_ids) + list(answer_ids)
        if len(input_ids) > self.context_limit:
            raise RuntimeError(
                "LoRA 样本超过完整上下文校验上限，代码不会截断 Prompt："
                f"index={idx}, prompt_tokens={len(prompt_ids)}, answer_tokens={len(answer_ids)}, "
                f"total={len(input_ids)}, context_limit={self.context_limit}。"
                "请检查 EVIDENCE_K_MAX 或提高 LORA_CONTEXT_LIMIT。"
            )

        labels = [-100] * len(prompt_ids) + list(answer_ids)
        if not labels or all(x == -100 for x in labels):
            raise RuntimeError(f"LoRA 样本 index={idx} 没有有效 answer token。")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


def _collate_lm(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].numel() for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].numel()
        input_ids[i, :n] = x["input_ids"]
        labels[i, :n] = x["labels"]
        attention_mask[i, :n] = x["attention_mask"]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def train_qwen_lora_on_time(
    processed_root: Path,
    time_points: Sequence[int],
    sample_time: int,
    anes_model,
    qwen_model_path: str,
    output_adapter_dir: Path,
    previous_adapter_dir: Optional[Path] = None,
    candidate_max_size: int = 256,
    train_pos_n: int = 5000,
    train_neg_n: int = 5000,
    seed: int = 2026,
    k_max: int = 8,
    context_limit: int = 8192,
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    show_progress: bool = True,
    torch_dtype: str = "auto",
    gradient_checkpointing: bool = False,
    fail_on_high_nan_ratio: bool = True,
    max_nan_ratio: float = 0.05,
    min_valid_steps: int = 1,
) -> Path:

    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
    from peft import LoraConfig, PeftModel, get_peft_model

    output_adapter_dir = Path(output_adapter_dir)
    output_adapter_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_dtype, resolved_dtype_name = _resolve_qwen_dtype(torch_dtype, device)
    print(f"[LoRA] Qwen 训练 dtype: {resolved_dtype_name}")
    model = AutoModelForCausalLM.from_pretrained(
        qwen_model_path,
        torch_dtype=resolved_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if hasattr(model, "config"):
        model.config.use_cache = False
    if previous_adapter_dir is not None and Path(previous_adapter_dir).exists():
        model = PeftModel.from_pretrained(model, str(previous_adapter_dir), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
    if bool(gradient_checkpointing) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if device != "cuda":
        model.to(device)
    model.train()


    qids = balanced_qid_sample(processed_root, sample_time, train_pos_n, train_neg_n, seed=seed)
    dataset = ANESTimeDataset(processed_root, time_points, sample_time, candidate_max_size, qid_filter=qids, use_weak=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=anes_collate)
    anes_model.eval().to(device)
    sample_by_qid = {int(s.qid): s for s in dataset.samples}

    examples: List[dict] = []
    for batch in tqdm(loader, desc=f"LoRA数据构造 | T={sample_time}", dynamic_ncols=True, leave=True, disable=not show_progress):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            sel = anes_model.select_evidence(batch, k_max=k_max, min_select=0)
        qid = int(batch["qid"][0].item())
        sample = sample_by_qid[qid]
        count = int(sel["selected_counts"][0].item())
        pos = sel["selected_candidate_positions"][0, :count].detach().cpu().tolist()
        cand_ids = batch["candidate_ids"][0].detach().cpu().tolist()
        selected_eids = [int(cand_ids[p]) for p in pos if p >= 0 and p < len(cand_ids) and cand_ids[p] >= 0]
        prompt = build_prompt(sample, selected_eids, dataset.entities, dataset.graph)
        prompt_text = format_chat_prompt(tokenizer, prompt)
        answer_text = "Yes" if int(sample.label) == 1 else "No"
        examples.append({"prompt_text": prompt_text, "answer_text": answer_text})

    lm_dataset = PromptAnswerDataset(examples, tokenizer, context_limit=context_limit)
    lm_loader = DataLoader(
        lm_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: _collate_lm(b, tokenizer.pad_token_id),
    )

    if len(lm_dataset) == 0:
        raise RuntimeError(f"Qwen LoRA 数据为空：T={sample_time}。")

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    grad_accum = max(1, int(grad_accum))
    expected_updates = max(1, math.ceil(len(lm_loader) * int(epochs) / grad_accum))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, expected_updates // 20),
        num_training_steps=expected_updates,
    )

    global_batch_step = 0
    valid_steps = 0
    optimizer_updates = 0
    skipped_nan = 0
    skipped_empty_label = 0
    running_loss = 0.0
    accum_steps = 0
    optimizer.zero_grad(set_to_none=True)

    def _optimizer_update() -> None:
        nonlocal optimizer_updates, accum_steps
        if accum_steps <= 0:
            return

        if accum_steps < grad_accum:
            scale = float(grad_accum) / float(accum_steps)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(scale)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_updates += 1
        accum_steps = 0

    for _epoch in range(int(epochs)):
        pbar = tqdm(
            lm_loader,
            desc=f"Qwen LoRA训练 | T={sample_time} | epoch={_epoch + 1}/{int(epochs)}",
            dynamic_ncols=True,
            leave=True,
            disable=not show_progress,
        )
        for batch in pbar:
            global_batch_step += 1
            model_device = next(model.parameters()).device
            batch = {k: v.to(model_device) for k, v in batch.items()}

            if int((batch["labels"] != -100).sum().detach().cpu()) == 0:
                skipped_empty_label += 1
                continue

            try:
                raw_loss = _supervised_token_loss(model, batch)
            except ValueError:
                skipped_empty_label += 1
                continue
            except FloatingPointError:

                skipped_nan += 1
                optimizer.zero_grad(set_to_none=True)
                accum_steps = 0
                pbar.set_postfix({
                    "loss": "nan-skip",
                    "batch": global_batch_step,
                    "updates": optimizer_updates,
                    "skip_nan": skipped_nan,
                    "skip_empty": skipped_empty_label,
                })
                continue

            (raw_loss / grad_accum).backward()
            running_loss += float(raw_loss.detach().cpu())
            valid_steps += 1
            accum_steps += 1
            if accum_steps >= grad_accum:
                _optimizer_update()

            pbar.set_postfix({
                "loss": f"{running_loss / max(1, valid_steps):.4f}",
                "batch": global_batch_step,
                "updates": optimizer_updates,
                "skip_nan": skipped_nan,
                "skip_empty": skipped_empty_label,
            })


        _optimizer_update()

    attempted_steps = max(1, valid_steps + skipped_nan + skipped_empty_label)
    nan_ratio = float(skipped_nan) / float(attempted_steps)
    stats = {
        "sample_time": int(sample_time),
        "num_examples": int(len(lm_dataset)),
        "epochs": int(epochs),
        "valid_steps": int(valid_steps),
        "optimizer_updates": int(optimizer_updates),
        "skipped_nan": int(skipped_nan),
        "skipped_empty_label": int(skipped_empty_label),
        "nan_ratio": float(nan_ratio),
        "torch_dtype": resolved_dtype_name,
        "context_limit": int(context_limit),
        "lr": float(lr),
        "grad_accum": int(grad_accum),
    }
    output_adapter_dir.mkdir(parents=True, exist_ok=True)
    with open(output_adapter_dir / "lora_train_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if (
        int(valid_steps) < int(min_valid_steps)
        or int(optimizer_updates) < 1
        or (bool(fail_on_high_nan_ratio) and nan_ratio > float(max_nan_ratio))
    ):

        msg = (
            f"Qwen LoRA 训练异常：valid_steps={valid_steps}, optimizer_updates={optimizer_updates}, skipped_nan={skipped_nan}, "
            f"nan_ratio={nan_ratio:.4f}。建议优先使用 LORA_TORCH_DTYPE='bfloat16' 或 'auto'，"
            f"并检查完整 Prompt token 数、LORA_CONTEXT_LIMIT 和 LORA_LR。"
        )
        try:

            for child in list(output_adapter_dir.iterdir()):
                if child.name != "lora_train_stats.json":
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(msg)

    model.save_pretrained(output_adapter_dir)
    tokenizer.save_pretrained(output_adapter_dir)
    return output_adapter_dir
