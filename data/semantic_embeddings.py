from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

try:
    import zstandard as zstd
except ImportError:
    zstd = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def open_text_maybe_zst(path: Path):

    if path.suffix == ".zst":
        if zstd is None:
            raise RuntimeError("读取 .zst 需要安装 zstandard：pip install zstandard")
        fh = path.open("rb")
        dctx = zstd.ZstdDecompressor()
        stream = dctx.stream_reader(fh)
        import io
        return io.TextIOWrapper(stream, encoding="utf-8")
    return path.open("r", encoding="utf-8", newline="")


def find_entities_file(root: Path) -> Path:

    p1 = root / "data" / "entities.tsv"
    p2 = root / "data" / "entities.tsv.zst"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"找不到实体表：{p1} 或 {p2}")


def load_entities(root: Path) -> Tuple[List[int], List[str], List[str], List[str]]:

    entities_file = find_entities_file(root)
    eids: List[int] = []
    raw_ids: List[str] = []
    names: List[str] = []
    descs: List[str] = []

    with open_text_maybe_zst(entities_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = int(row["eid"])
            eids.append(eid)
            raw_ids.append(row.get("raw_id", ""))
            names.append(row.get("name", "") or row.get("raw_id", ""))
            descs.append(row.get("desc", "") or "")

    order = np.argsort(np.asarray(eids))
    eids = [eids[i] for i in order]
    raw_ids = [raw_ids[i] for i in order]
    names = [names[i] for i in order]
    descs = [descs[i] for i in order]

    expected = list(range(len(eids)))
    if eids != expected:
        raise ValueError("entities.tsv 中 eid 必须从 0 到 num_entities-1 连续排列。")
    return eids, raw_ids, names, descs


def build_entity_texts(names: List[str], descs: List[str]) -> List[str]:

    texts: List[str] = []
    for name, desc in zip(names, descs):
        name = (name or "").strip()
        desc = (desc or "").strip()
        if name and desc:
            text = f"{name}. {desc}"
        elif name:
            text = name
        else:
            text = desc
        if not text:
            text = "unknown biomedical entity"
        texts.append(text)
    return texts


def mean_pooling(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:

    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = torch.sum(last_hidden * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-6)
    return summed / counts


def main(cfg: dict) -> None:
    set_seed(int(cfg["SEED"]))

    root = Path(cfg["PROCESSED_ROOT"]).expanduser().resolve()
    output_path = root / cfg["OUTPUT_FILE"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] 读取实体表：{root / 'data'}")
    eids, raw_ids, names, descs = load_entities(root)
    texts = build_entity_texts(names, descs)
    print(f"      实体数量：{len(texts)}")

    print(f"[2/5] 加载 PubMedBERT 模型：{cfg['MODEL_NAME_OR_PATH']}")
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device(str(cfg["DEVICE"]))
    tokenizer = AutoTokenizer.from_pretrained(str(cfg["MODEL_NAME_OR_PATH"]))
    model = AutoModel.from_pretrained(str(cfg["MODEL_NAME_OR_PATH"]))
    model.to(device)
    model.eval()
    print(f"      使用设备：{device}")

    batch_size = int(cfg["BATCH_SIZE"])
    max_length = int(cfg["MAX_LENGTH"])
    normalize = bool(cfg["NORMALIZE"])

    all_embs: List[np.ndarray] = []
    print("[3/5] 开始编码实体文本。")
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            emb = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
            if normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.detach().cpu().numpy().astype(np.float32))

            batch_id = start // batch_size + 1
            if batch_id % int(cfg["LOG_EVERY"]) == 0 or start + batch_size >= len(texts):
                print(f"      已编码 {min(start + batch_size, len(texts))}/{len(texts)}")

    emb_np = np.concatenate(all_embs, axis=0)
    print(f"[4/5] 向量 shape = {emb_np.shape}")
    if emb_np.shape[0] != len(eids):
        raise RuntimeError("语义向量行数与实体数量不一致。")

    if cfg["OUTPUT_DTYPE"] == "float16":
        emb_to_save = emb_np.astype(np.float16)
    else:
        emb_to_save = emb_np.astype(np.float32)

    np.save(output_path, emb_to_save)

    meta = {
        "embedding_type": "global_static_pubmedbert_mean_pooling",
        "model_name_or_path": str(cfg["MODEL_NAME_OR_PATH"]),
        "input_text": "name + '. ' + desc",
        "shape": list(emb_to_save.shape),
        "dtype": str(emb_to_save.dtype),
        "normalize": normalize,
        "max_length": max_length,
        "row_index": "eid",
        "note": "全局静态语义向量；没有按时间过滤 desc，适合快速测试。严格时间实验建议构建按 context_time 切分的语义向量。",
    }
    meta_path = output_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[5/5] 已保存语义向量：{output_path}")
    print(f"      已保存元信息：{meta_path}")
    print("Done.")
