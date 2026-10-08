from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def find_existing_file(root: Path, relative_without_zst: str) -> Path:

    p_plain = root / relative_without_zst
    p_zst = root / f"{relative_without_zst}.zst"
    if p_plain.exists():
        return p_plain
    if p_zst.exists():
        return p_zst
    raise FileNotFoundError(f"找不到文件：{p_plain} 或 {p_zst}")


def read_num_entities(root: Path, entity_file: str) -> int:

    entity_path = find_existing_file(root, entity_file)
    eids: List[int] = []
    with open_text_maybe_zst(entity_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "eid" not in (reader.fieldnames or []):
            raise ValueError(f"实体表缺少 eid 列：{entity_path}")
        for row in reader:
            eids.append(int(row["eid"]))

    if not eids:
        raise ValueError(f"实体表为空：{entity_path}")
    eids_sorted = sorted(eids)
    expected = list(range(len(eids_sorted)))
    if eids_sorted != expected:
        raise ValueError("entities.tsv 中 eid 必须从 0 连续编号，否则无法保证向量行号与实体顺序一致。")
    return len(eids_sorted)


def read_edges(root: Path, time: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    edge_file = find_existing_file(root, f"data/graph/{time}.edges.tsv")
    srcs: List[int] = []
    dsts: List[int] = []
    freqs: List[float] = []
    with open_text_maybe_zst(edge_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s = int(row["src"])
            d = int(row["dst"])
            if s == d:
                continue
            srcs.append(s)
            dsts.append(d)
            freqs.append(float(row.get("freq", 1.0) or 1.0))
    return np.asarray(srcs, dtype=np.int64), np.asarray(dsts, dtype=np.int64), np.asarray(freqs, dtype=np.float32)


def build_norm_adj_and_stats(
    num_nodes: int,
    src: np.ndarray,
    dst: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:

    degree = np.zeros(num_nodes, dtype=np.int64)
    if len(src) > 0:
        np.add.at(degree, src, 1)
        np.add.at(degree, dst, 1)
    appeared_mask = (degree > 0).astype(np.uint8)
    active_nodes = np.where(appeared_mask > 0)[0].astype(np.int64)

    if len(src) == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float32, device=device)
        adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()
        return adj, degree, appeared_mask, active_nodes

    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    deg_float = np.bincount(row, minlength=num_nodes).astype(np.float32)
    deg_float[deg_float == 0] = 1.0
    vals = 1.0 / deg_float[row]

    indices = torch.tensor(np.stack([row, col], axis=0), dtype=torch.long, device=device)
    values = torch.tensor(vals, dtype=torch.float32, device=device)
    adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()
    return adj, degree, appeared_mask, active_nodes


class MeanSAGELayer(nn.Module):


    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_mean: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(adj_mean, x)
        out = self.self_linear(x) + self.neigh_linear(neigh)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class GraphSAGEEncoder(nn.Module):


    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, dropout: float):
        super().__init__()
        if num_layers < 1:
            raise ValueError("NUM_LAYERS 至少为 1")
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            MeanSAGELayer(dims[i], dims[i + 1], dropout=dropout if i < num_layers - 1 else 0.0)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, adj_mean: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, adj_mean)
        return h


def make_edge_set(src: np.ndarray, dst: np.ndarray) -> Set[Tuple[int, int]]:

    edges: Set[Tuple[int, int]] = set()
    for s, d in zip(src.tolist(), dst.tolist()):
        if s == d:
            continue
        a, b = (s, d) if s < d else (d, s)
        edges.add((a, b))
    return edges


def sample_negative_edges(
    node_pool: np.ndarray,
    num_samples: int,
    existing: Set[Tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:

    if len(node_pool) < 2:
        raise ValueError("负采样节点池少于 2 个节点，无法采样负边。")

    neg: List[Tuple[int, int]] = []
    attempts = 0
    pool = np.asarray(node_pool, dtype=np.int64)
    while len(neg) < num_samples:
        attempts += 1
        need = num_samples - len(neg)
        draw = max(need * 3, 2048)
        a = np.random.choice(pool, size=draw, replace=True)
        b = np.random.choice(pool, size=draw, replace=True)
        for u, v in zip(a.tolist(), b.tolist()):
            if u == v:
                continue
            x, y = (u, v) if u < v else (v, u)
            if (x, y) not in existing:
                neg.append((x, y))
                if len(neg) >= num_samples:
                    break
        if attempts > 100 and len(neg) == 0:
            raise RuntimeError("负采样失败：图可能过于稠密，或 active node 数量过少。")
    arr = np.asarray(neg[:num_samples], dtype=np.int64)
    return torch.tensor(arr, dtype=torch.long, device=device)


def batched_indices(num_items: int, batch_size: int, shuffle: bool = True) -> Iterable[np.ndarray]:
    idx = np.arange(num_items)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, num_items, batch_size):
        yield idx[start:start + batch_size]


def create_optimizer(model: nn.Module, cfg: Dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=float(cfg["LR"]), weight_decay=float(cfg["WEIGHT_DECAY"]))


def save_checkpoint(path: Path, model: nn.Module, cfg: Dict, time: int, in_dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "time": int(time),
        "state_dict": model.state_dict(),
        "in_dim": int(in_dim),
        "hidden_dim": int(cfg["HIDDEN_DIM"]),
        "out_dim": int(cfg["OUT_DIM"]),
        "num_layers": int(cfg["NUM_LAYERS"]),
        "dropout": float(cfg["DROPOUT"]),
        "incremental_training": bool(cfg["INCREMENTAL_TRAINING"]),
    }, path)


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> None:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])


def train_one_snapshot(
    *,
    time: int,
    model: GraphSAGEEncoder,
    x_all: torch.Tensor,
    src: np.ndarray,
    dst: np.ndarray,
    freq: np.ndarray,
    epochs: int,
    cfg: Dict,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    num_nodes, _ = x_all.shape
    print(f"\n[GraphSAGE] time={time}, num_nodes={num_nodes}, num_edges={len(src)}, epochs={epochs}")

    adj_mean, degree, appeared_mask, active_nodes = build_norm_adj_and_stats(num_nodes, src, dst, device=device)
    edge_set = make_edge_set(src, dst)

    if str(cfg["NEGATIVE_SAMPLE_NODE_SCOPE"]).lower() == "active" and len(active_nodes) >= 2:
        neg_node_pool = active_nodes
        print(f"      负采样范围：active nodes = {len(active_nodes)}")
    else:
        neg_node_pool = np.arange(num_nodes, dtype=np.int64)
        print(f"      负采样范围：all nodes = {num_nodes}")


    if len(src) == 0:
        print("      当前图没有边，跳过训练，仅输出当前模型的 self-only / 空邻接结果。")
        model.eval()
        with torch.no_grad():
            h = model(x_all, adj_mean)
            if bool(cfg["NORMALIZE_OUTPUT"]):
                h = F.normalize(h, p=2, dim=-1)
            out = h.detach().cpu().numpy().astype(np.float32)
        return out, degree, appeared_mask

    optimizer = create_optimizer(model, cfg)

    pos_edges_np = np.stack([src, dst], axis=1).astype(np.int64)
    pos_freq_np = freq.astype(np.float32)
    batch_size = int(cfg["EDGE_BATCH_SIZE"])
    neg_ratio = float(cfg["NEGATIVE_RATIO"])

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for idx in batched_indices(len(pos_edges_np), batch_size=batch_size, shuffle=True):
            pos_edges = torch.tensor(pos_edges_np[idx], dtype=torch.long, device=device)
            num_pos = pos_edges.size(0)
            num_neg = max(1, int(num_pos * neg_ratio))
            neg_edges = sample_negative_edges(neg_node_pool, num_neg, edge_set, device=device)


            h = model(x_all, adj_mean)
            h = F.normalize(h, p=2, dim=-1)

            pos_score = (h[pos_edges[:, 0]] * h[pos_edges[:, 1]]).sum(dim=-1)
            neg_score = (h[neg_edges[:, 0]] * h[neg_edges[:, 1]]).sum(dim=-1)

            logits = torch.cat([pos_score, neg_score], dim=0)
            labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)

            if bool(cfg["USE_FREQ_AS_POS_WEIGHT"]):
                w_pos_np = np.log1p(pos_freq_np[idx])
                w_pos_np = np.clip(w_pos_np, 1.0, float(cfg["MAX_FREQ_WEIGHT"])).astype(np.float32)
                weights = torch.cat([
                    torch.tensor(w_pos_np, dtype=torch.float32, device=device),
                    torch.ones_like(neg_score),
                ], dim=0)
                loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item()) * int(labels.numel())
            total_count += int(labels.numel())

        if epoch % int(cfg["LOG_EVERY_EPOCH"]) == 0 or epoch == int(epochs):
            print(f"      epoch {epoch:03d}/{epochs}, loss={total_loss / max(total_count, 1):.6f}")

    model.eval()
    with torch.no_grad():
        h = model(x_all, adj_mean)
        if bool(cfg["NORMALIZE_OUTPUT"]):
            h = F.normalize(h, p=2, dim=-1)
        out = h.detach().cpu().numpy().astype(np.float32)
    return out, degree, appeared_mask


def main(cfg: dict) -> None:
    set_seed(int(cfg["SEED"]))

    root = Path(cfg["PROCESSED_ROOT"]).expanduser().resolve()
    sem_path = root / cfg["SEMANTIC_EMB_FILE"]
    out_dir = root / cfg["STRUCT_EMB_DIR"]
    ckpt_dir = root / cfg["MODEL_CKPT_DIR"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not sem_path.exists():
        raise FileNotFoundError(f"找不到语义向量文件：{sem_path}。请先运行 python data/DatasetPreprocessing.py")

    print(f"[1/4] 读取实体表并检查 eid 顺序：{root / cfg['ENTITY_FILE']}")
    num_entities = read_num_entities(root, str(cfg["ENTITY_FILE"]))
    print(f"      实体数量：{num_entities}")

    print(f"[2/4] 读取语义向量：{sem_path}")
    sem = np.load(sem_path).astype(np.float32)
    if sem.shape[0] != num_entities:
        raise ValueError(
            f"语义向量行数与 entities.tsv 不一致：sem.shape[0]={sem.shape[0]}, num_entities={num_entities}。"
            "请确认语义向量是在当前 processed dataset 上生成的。"
        )
    print(f"      semantic shape = {sem.shape}, dtype=float32 for training")

    device = resolve_device(str(cfg["DEVICE"]))
    print(f"[3/4] 使用设备：{device}")
    x_all = torch.tensor(sem, dtype=torch.float32, device=device)

    x_all = F.normalize(x_all, p=2, dim=-1)

    in_dim = int(sem.shape[1])
    times = [int(t) for t in cfg["CONTEXT_TIME_POINTS"]]

    model: Optional[GraphSAGEEncoder] = None
    if bool(cfg["INCREMENTAL_TRAINING"]):
        print("[4/4] 开始逐时间点增量训练 GraphSAGE。")
        model = GraphSAGEEncoder(
            in_dim=in_dim,
            hidden_dim=int(cfg["HIDDEN_DIM"]),
            out_dim=int(cfg["OUT_DIM"]),
            num_layers=int(cfg["NUM_LAYERS"]),
            dropout=float(cfg["DROPOUT"]),
        ).to(device)
    else:
        print("[4/4] 开始逐时间点独立训练 GraphSAGE。")

    summary = {
        "embedding_type": "graphsage_link_reconstruction",
        "training_mode": "incremental_warm_start" if bool(cfg["INCREMENTAL_TRAINING"]) else "from_scratch_per_snapshot",
        "input_feature": str(cfg["SEMANTIC_EMB_FILE"]),
        "context_time_points": times,
        "semantic_dim": in_dim,
        "out_dim": int(cfg["OUT_DIM"]),
        "hidden_dim": int(cfg["HIDDEN_DIM"]),
        "num_layers": int(cfg["NUM_LAYERS"]),
        "dropout": float(cfg["DROPOUT"]),
        "row_index": "eid，严格对应 data/entities.tsv 中的 eid",
        "all_entity_rows_saved": True,
        "usage": "sample/T 使用 prev(T) 的结构向量，例如 sample/2000 使用 struct_emb/1990.graphsage.f16.npy",
        "negative_sample_node_scope": str(cfg["NEGATIVE_SAMPLE_NODE_SCOPE"]),
        "files": {},
    }

    for idx_time, time in enumerate(times):
        out_path = out_dir / f"{time}.graphsage.f16.npy"
        degree_path = out_dir / f"{time}.degree.npy"
        mask_path = out_dir / f"{time}.appeared_mask.npy"
        ckpt_path = ckpt_dir / f"{time}.graphsage.pt"


        if not bool(cfg["INCREMENTAL_TRAINING"]):
            model = GraphSAGEEncoder(
                in_dim=in_dim,
                hidden_dim=int(cfg["HIDDEN_DIM"]),
                out_dim=int(cfg["OUT_DIM"]),
                num_layers=int(cfg["NUM_LAYERS"]),
                dropout=float(cfg["DROPOUT"]),
            ).to(device)

        assert model is not None

        if bool(cfg["SKIP_IF_EXISTS"]) and out_path.exists():
            print(f"\n[GraphSAGE] time={time} 已存在，跳过：{out_path}")
            if bool(cfg["INCREMENTAL_TRAINING"]) and bool(cfg["LOAD_CKPT_WHEN_SKIP"]) and ckpt_path.exists():
                print(f"      加载 checkpoint 以便后续年份 warm-start：{ckpt_path}")
                load_checkpoint(ckpt_path, model, device)
            elif bool(cfg["INCREMENTAL_TRAINING"]):
                print("      警告：跳过了向量文件，但没有加载对应 checkpoint，后续年份可能无法严格继承该年份参数。")
            summary["files"][str(time)] = {
                "embedding": str(out_path.relative_to(root)),
                "degree": str(degree_path.relative_to(root)) if degree_path.exists() else None,
                "appeared_mask": str(mask_path.relative_to(root)) if mask_path.exists() else None,
                "checkpoint": str(ckpt_path.relative_to(root)) if ckpt_path.exists() else None,
            }
            continue

        src, dst, freq = read_edges(root, time)

        if bool(cfg["INCREMENTAL_TRAINING"]):
            epochs = int(cfg["FIRST_TIME_EPOCHS"]) if idx_time == 0 else int(cfg["INCREMENTAL_EPOCHS"])
        else:
            epochs = int(cfg["EPOCHS_WHEN_TRAIN_FROM_SCRATCH"])

        emb, degree, appeared_mask = train_one_snapshot(
            time=time,
            model=model,
            x_all=x_all,
            src=src,
            dst=dst,
            freq=freq,
            epochs=epochs,
            cfg=cfg,
            device=device,
        )

        emb_to_save = emb.astype(np.float16 if cfg["OUTPUT_DTYPE"] == "float16" else np.float32)
        if emb_to_save.shape[0] != num_entities:
            raise RuntimeError("内部错误：结构向量行数不是全体实体数量。")

        np.save(out_path, emb_to_save)
        np.save(degree_path, degree.astype(np.int32))
        np.save(mask_path, appeared_mask.astype(np.uint8))

        if bool(cfg["SAVE_MODEL_CHECKPOINT"]):
            save_checkpoint(ckpt_path, model, cfg, time=time, in_dim=in_dim)
            ckpt_rel = str(ckpt_path.relative_to(root))
        else:
            ckpt_rel = None

        summary["files"][str(time)] = {
            "embedding": str(out_path.relative_to(root)),
            "degree": str(degree_path.relative_to(root)),
            "appeared_mask": str(mask_path.relative_to(root)),
            "checkpoint": ckpt_rel,
            "shape": list(emb_to_save.shape),
            "dtype": str(emb_to_save.dtype),
            "epochs": epochs,
        }
        print(f"      已保存结构向量：{out_path} shape={emb_to_save.shape} dtype={emb_to_save.dtype}")
        print(f"      已保存 degree：{degree_path}")
        print(f"      已保存 appeared_mask：{mask_path}")
        if ckpt_rel is not None:
            print(f"      已保存 checkpoint：{ckpt_path}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    meta_path = out_dir / "graphsage_meta.json"
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存 GraphSAGE 元信息：{meta_path}")
    print("Done.")
