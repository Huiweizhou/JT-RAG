from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def read_json_safe(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {} if default is None else dict(default)
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return obj
    except Exception as exc:
        print(f"[Resume] 读取 JSON 失败，将使用默认值：{path} | {exc}")
    return {} if default is None else dict(default)


def write_json_atomic(path: Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def init_state(path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    state = read_json_safe(path, default={})
    if not state:
        state = {
            "version": 1,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "output_dir": str(Path(cfg["OUTPUT_DIR"]).resolve()),
            "last_anes_ckpt": None,
            "active_lora_adapter": None,
            "completed_stages": {},
            "running_stage": None,
        }
        write_json_atomic(path, state)
    state.setdefault("version", 1)
    state.setdefault("completed_stages", {})
    state.setdefault("last_anes_ckpt", None)
    state.setdefault("active_lora_adapter", None)
    state.setdefault("running_stage", None)
    return state


def save_state(path: Path, state: Dict[str, Any]) -> None:
    state = dict(state)
    state["updated_at"] = utc_now()
    write_json_atomic(path, state)


def stage_key(kind: str, time: int, round_idx: Optional[int] = None) -> str:
    if round_idx is None:
        return f"{kind}:T{int(time)}"
    return f"{kind}:T{int(time)}:R{int(round_idx)}"


def is_completed(state: Dict[str, Any], key: str) -> bool:
    return key in state.get("completed_stages", {})


def get_stage_meta(state: Dict[str, Any], key: str) -> Dict[str, Any]:
    return dict(state.get("completed_stages", {}).get(key, {}))


def mark_running(state_path: Path, state: Dict[str, Any], key: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state["running_stage"] = {
        "key": key,
        "started_at": utc_now(),
        "meta": meta or {},
    }
    save_state(state_path, state)
    return state


def mark_completed(state_path: Path, state: Dict[str, Any], key: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state.setdefault("completed_stages", {})[key] = {
        "completed_at": utc_now(),
        **(meta or {}),
    }
    if state.get("running_stage", {}).get("key") == key:
        state["running_stage"] = None
    if meta:
        if meta.get("ckpt_path"):
            state["last_anes_ckpt"] = str(meta["ckpt_path"])
        if meta.get("adapter_path"):
            state["active_lora_adapter"] = str(meta["adapter_path"])
    save_state(state_path, state)
    return state


def is_valid_lora_adapter(path: Optional[Path]) -> bool:
    if path is None:
        return False
    path = Path(path)
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "adapter_config.json").exists():
        return False
    has_weight = any((path / name).exists() for name in ["adapter_model.safetensors", "adapter_model.bin"])
    return bool(has_weight)


def remove_incomplete_lora_adapter(path: Path) -> None:
    path = Path(path)
    if path.exists() and path.is_dir() and not is_valid_lora_adapter(path):
        print(f"[Resume] 删除未完成的 LoRA adapter 目录：{path}")
        shutil.rmtree(path, ignore_errors=True)


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    dev = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(dev)


def save_anes_checkpoint(
    path: Path,
    model,
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    sample_time: int,
    stage: str,
    epoch: Optional[int] = None,
    stage_key_value: Optional[str] = None,
    completed: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": cfg,
        "sample_time": int(sample_time),
        "stage": stage,
        "epoch": None if epoch is None else int(epoch),
        "stage_key": stage_key_value,
        "completed": bool(completed),
        "saved_at": utc_now(),
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_anes_checkpoint(
    path: Path,
    model,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    path = Path(path)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=strict)
    model.to(device)
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            move_optimizer_to_device(optimizer, device)
        except Exception as exc:
            print(f"[Resume] optimizer_state 加载失败，将继续使用新 optimizer：{exc}")
    print(f"[Resume] 已加载 ANES checkpoint：{path}")
    return ckpt


def find_latest_epoch_checkpoint(ckpt_dir: Path, prefix: str) -> Tuple[Optional[Path], int]:

    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None, 0
    pattern = re.compile(re.escape(prefix) + r"_epoch(\d+)\.pt$")
    best_path: Optional[Path] = None
    best_epoch = 0
    for path in ckpt_dir.glob(f"{prefix}_epoch*.pt"):
        m = pattern.search(path.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = path
    return best_path, best_epoch


def maybe_load_completed_stage_ckpt(
    state: Dict[str, Any],
    key: str,
    model,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    load_model: bool = True,
) -> Optional[Path]:

    if not load_model or not is_completed(state, key):
        return None
    meta = get_stage_meta(state, key)
    ckpt_path = meta.get("ckpt_path")
    if ckpt_path and Path(ckpt_path).exists():
        load_anes_checkpoint(Path(ckpt_path), model, optimizer, device=device)
        return Path(ckpt_path)
    return None
