from __future__ import annotations

import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F

from .prompt_builder import format_chat_prompt


def _resolve_qwen_dtype(torch_dtype: str, device: str):
    name = str(torch_dtype or "auto").lower()
    if device != "cuda":
        return torch.float32
    if name == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    return torch.float16


class QwenYesNoRunner:


    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        torch_dtype: str = "auto",
        adapter_path: Optional[str] = None,
        context_limit: int = 8192,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self.context_limit = int(context_limit)
        if self.context_limit <= 0:
            raise ValueError("context_limit 必须为正整数。")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = _resolve_qwen_dtype(torch_dtype, device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
        if device != "cuda":
            self.model.to(device)
        self.model.eval()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _format_chat(self, prompt: str) -> str:
        return format_chat_prompt(self.tokenizer, prompt)


    def _completion_log_probs(
        self,
        prompt_texts: Sequence[str],
        completions: Sequence[str],
        score_batch_size: int = 8,
    ) -> List[tuple[float, bool]]:

        if len(prompt_texts) != len(completions):
            raise ValueError("prompt_texts 与 completions 数量不一致。")
        if not prompt_texts:
            return []

        all_scores: List[tuple[float, bool]] = []
        batch_size = max(1, int(score_batch_size))
        for start in range(0, len(prompt_texts), batch_size):
            ps = list(prompt_texts[start : start + batch_size])
            cs = list(completions[start : start + batch_size])
            encoded_items: List[tuple[List[int], int, int, bool]] = []
            max_len = 0
            for local_idx, (prompt, completion) in enumerate(zip(ps, cs)):
                prompt_ids = self.tokenizer(
                    prompt, add_special_tokens=False, truncation=False
                )["input_ids"]
                comp_ids = self.tokenizer(
                    completion, add_special_tokens=False, truncation=False
                )["input_ids"]
                if len(comp_ids) == 0:
                    comp_ids = self.tokenizer(
                        " " + completion.strip(), add_special_tokens=False, truncation=False
                    )["input_ids"]
                if len(comp_ids) == 0:
                    encoded_items.append(([], 0, 0, False))
                    continue

                input_ids = list(prompt_ids) + list(comp_ids)
                global_idx = start + local_idx
                if len(input_ids) > self.context_limit:
                    raise RuntimeError(
                        "Qwen reward/eval scorer 输入超过完整上下文校验上限，代码不会截断 Prompt："
                        f"item_index={global_idx}, prompt_tokens={len(prompt_ids)}, "
                        f"completion_tokens={len(comp_ids)}, total={len(input_ids)}, "
                        f"context_limit={self.context_limit}。"
                    )
                encoded_items.append((input_ids, len(prompt_ids), len(comp_ids), True))
                max_len = max(max_len, len(input_ids))

            if max_len <= 0:
                all_scores.extend([(-100.0, False) for _ in encoded_items])
                continue

            input_ids_tensor = torch.full(
                (len(encoded_items), max_len), int(self.tokenizer.pad_token_id), dtype=torch.long
            )
            attention_mask = torch.zeros((len(encoded_items), max_len), dtype=torch.long)
            for i, (ids, _prompt_len, _comp_len, valid) in enumerate(encoded_items):
                if not valid:
                    continue
                n = len(ids)
                input_ids_tensor[i, :n] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, :n] = 1

            input_ids_tensor = input_ids_tensor.to(self.device)
            attention_mask = attention_mask.to(self.device)
            with torch.inference_mode():
                logits = self.model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits

            for i, (_ids, prompt_len, comp_len, valid_item) in enumerate(encoded_items):
                if not valid_item:
                    all_scores.append((-100.0, False))
                    continue
                token_scores: List[float] = []
                valid = True
                for j in range(comp_len):
                    token_pos = prompt_len + j
                    pred_pos = token_pos - 1
                    if pred_pos < 0 or token_pos >= input_ids_tensor.size(1):
                        valid = False
                        break
                    one_logits = logits[i, pred_pos, :]
                    if not torch.isfinite(one_logits).all():
                        valid = False
                        break
                    token_id = int(input_ids_tensor[i, token_pos].item())
                    lp = F.log_softmax(one_logits.float(), dim=-1)[token_id]
                    if not torch.isfinite(lp):
                        valid = False
                        break
                    token_scores.append(float(lp.detach().cpu()))
                if valid and len(token_scores) == comp_len and token_scores:
                    all_scores.append((float(sum(token_scores)), True))
                else:
                    all_scores.append((-100.0, False))
        return all_scores

    def score_yes_no_batch(
        self,
        prompts: Sequence[str],
        labels: Optional[Sequence[int]] = None,
        score_batch_size: int = 8,
    ) -> List[dict]:

        if not prompts:
            return []
        if labels is not None and len(labels) != len(prompts):
            raise ValueError("labels 与 prompts 数量不一致。")

        prompt_texts = [self._format_chat(p) for p in prompts]
        yes_scores = self._completion_log_probs(
            prompt_texts, ["Yes"] * len(prompt_texts), score_batch_size
        )
        no_scores = self._completion_log_probs(
            prompt_texts, ["No"] * len(prompt_texts), score_batch_size
        )
        if len(yes_scores) != len(prompts) or len(no_scores) != len(prompts):
            raise RuntimeError("Qwen Yes/No scorer 返回数量异常。")

        records: List[dict] = []
        for i, ((ly, vy), (ln, vn)) in enumerate(zip(yes_scores, no_scores)):
            score_valid = bool(vy and vn and math.isfinite(ly) and math.isfinite(ln))
            if score_valid:
                m = max(ly, ln)
                log_denom = m + math.log(math.exp(ly - m) + math.exp(ln - m))
                log_p_yes_binary = ly - log_denom
                log_p_no_binary = ln - log_denom
                p_yes = math.exp(log_p_yes_binary)
                p_no = math.exp(log_p_no_binary)
                probability_pred = 1 if ly >= ln else 0
                if labels is None:
                    log_prob_true_raw = None
                    log_prob_true_binary = None
                    margin_true = None
                else:
                    label = int(labels[i])
                    log_prob_true_raw = ly if label == 1 else ln
                    log_prob_true_binary = log_p_yes_binary if label == 1 else log_p_no_binary
                    margin_true = (ly - ln) if label == 1 else (ln - ly)
            else:
                ly = ln = -100.0
                p_yes = p_no = 0.5
                log_p_yes_binary = log_p_no_binary = math.log(0.5)
                probability_pred = -1
                log_prob_true_raw = None if labels is None else -100.0
                log_prob_true_binary = None if labels is None else -100.0
                margin_true = None if labels is None else -200.0

            records.append(
                {
                    "log_prob_yes": float(ly),
                    "log_prob_no": float(ln),
                    "p_yes": float(p_yes),
                    "p_no": float(p_no),
                    "log_p_yes_binary": float(log_p_yes_binary),
                    "log_p_no_binary": float(log_p_no_binary),
                    "log_prob_true_raw": (
                        None if log_prob_true_raw is None else float(log_prob_true_raw)
                    ),
                    "log_prob_true_binary": (
                        None if log_prob_true_binary is None else float(log_prob_true_binary)
                    ),

                    "log_prob_true": (
                        None if log_prob_true_binary is None else float(log_prob_true_binary)
                    ),
                    "margin_true": None if margin_true is None else float(margin_true),
                    "probability_pred": int(probability_pred),
                    "score_valid": int(score_valid),
                }
            )
        return records
