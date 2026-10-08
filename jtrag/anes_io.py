from __future__ import annotations

import csv
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Set

import numpy as np


class _ClosingTextIOWrapper(io.TextIOWrapper):


    def __init__(self, buffer, raw_fh, *args, **kwargs):
        super().__init__(buffer, *args, **kwargs)
        self._raw_fh = raw_fh

    def close(self):
        try:
            super().close()
        finally:
            try:
                self._raw_fh.close()
            except Exception:
                pass


def open_text(path: Path, mode: str = "rt", encoding: str = "utf-8"):

    path = Path(path)
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("读取 .zst 需要安装 zstandard：pip install zstandard") from exc

        if "b" in mode:
            return open(path, mode)

        binary_mode = mode.replace("t", "").replace("r", "rb").replace("w", "wb")
        raw_fh = open(path, binary_mode)
        if "r" in mode:
            stream = zstd.ZstdDecompressor().stream_reader(raw_fh)
        else:
            stream = zstd.ZstdCompressor(level=3).stream_writer(raw_fh)
        return _ClosingTextIOWrapper(stream, raw_fh, encoding=encoding)

    return open(path, mode, encoding=encoding, newline="")


def prefer_zst_or_tsv(path_without_zst: Path) -> Path:

    path_without_zst = Path(path_without_zst)
    zst_path = Path(str(path_without_zst) + ".zst")
    if zst_path.exists():
        return zst_path
    if path_without_zst.exists():
        return path_without_zst

    return zst_path


def read_tsv(path: Path) -> Iterator[Dict[str, str]]:

    with open_text(path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


@dataclass(frozen=True)
class EntityRecord:
    eid: int
    raw_id: str
    name: str
    desc: str


@dataclass(frozen=True)
class SampleRecord:
    qid: int
    src: int
    dst: int
    label: int


@dataclass(frozen=True)
class EdgeInfo:
    src: int
    dst: int
    freq: int
    first_time: int


class GraphSnapshot:


    def __init__(self, edges: Iterable[EdgeInfo]):
        self.adj: Dict[int, Dict[int, EdgeInfo]] = defaultdict(dict)
        self.edges: List[EdgeInfo] = []
        for e in edges:
            s, d = (int(e.src), int(e.dst))
            if s > d:
                s, d = d, s
            info = EdgeInfo(s, d, int(e.freq), int(e.first_time))
            self.edges.append(info)
            self.adj[s][d] = info
            self.adj[d][s] = info

    @classmethod
    def from_file(cls, path: Path) -> "GraphSnapshot":
        edges: List[EdgeInfo] = []
        for row in read_tsv(path):
            edges.append(
                EdgeInfo(
                    src=int(row["src"]),
                    dst=int(row["dst"]),
                    freq=int(row["freq"]),
                    first_time=int(row["first_time"]),
                )
            )
        return cls(edges)

    def neighbors(self, eid: int) -> Set[int]:
        return set(self.adj.get(int(eid), {}).keys())

    def edge(self, a: int, b: int) -> Optional[EdgeInfo]:
        return self.adj.get(int(a), {}).get(int(b))

    def degree(self, eid: int) -> int:
        return len(self.adj.get(int(eid), {}))

    def candidate_set(self, src: int, dst: int) -> Set[int]:

        src = int(src)
        dst = int(dst)
        return (self.neighbors(src) | self.neighbors(dst)) - {src, dst}


def load_entities(processed_root: Path) -> Dict[int, EntityRecord]:

    path = prefer_zst_or_tsv(Path(processed_root) / "data" / "entities.tsv")
    entities: Dict[int, EntityRecord] = {}
    for row in read_tsv(path):
        eid = int(row["eid"])
        entities[eid] = EntityRecord(
            eid=eid,
            raw_id=row.get("raw_id", ""),
            name=row.get("name", ""),
            desc=row.get("desc", ""),
        )
    return entities


def load_samples(processed_root: Path, sample_time: int, sample_file: Optional[Path] = None) -> List[SampleRecord]:

    path = prefer_zst_or_tsv(
        Path(sample_file) if sample_file is not None
        else Path(processed_root) / "sample" / f"{int(sample_time)}.tsv"
    )
    samples: List[SampleRecord] = []
    for row in read_tsv(path):
        samples.append(
            SampleRecord(
                qid=int(row["qid"]),
                src=int(row["src"]),
                dst=int(row["dst"]),
                label=int(row["label"]),
            )
        )
    return samples


def load_weak_map(processed_root: Path, sample_time: int) -> Dict[int, Set[int]]:
    path = prefer_zst_or_tsv(Path(processed_root) / "sample" / f"{int(sample_time)}.weak.tsv")
    weak: Dict[int, Set[int]] = defaultdict(set)
    if not path.exists():
        return weak
    for row in read_tsv(path):
        weak[int(row["qid"])].add(int(row["evidence"]))
    return weak


def load_graph(processed_root: Path, context_time: int) -> GraphSnapshot:
    path = prefer_zst_or_tsv(Path(processed_root) / "data" / "graph" / f"{int(context_time)}.edges.tsv")
    return GraphSnapshot.from_file(path)


def load_semantic_embeddings(processed_root: Path, mmap_mode: str = "r") -> np.ndarray:
    path = Path(processed_root) / "data" / "entity_sem_emb.f16.npy"
    if not path.exists():
        raise FileNotFoundError(f"找不到语义向量：{path}")
    return np.load(path, mmap_mode=mmap_mode)


def load_struct_embeddings(processed_root: Path, context_time: int, mmap_mode: str = "r") -> np.ndarray:
    path = Path(processed_root) / "data" / "struct_emb" / f"{int(context_time)}.graphsage.f16.npy"
    if not path.exists():
        raise FileNotFoundError(f"找不到结构向量：{path}")
    return np.load(path, mmap_mode=mmap_mode)


def previous_time(time_points: List[int], sample_time: int) -> int:

    sample_time = int(sample_time)
    idx = time_points.index(sample_time)
    if idx <= 0:
        raise ValueError(f"{sample_time} 没有上一个时间点，不能严格构造历史上下文。")
    return int(time_points[idx - 1])
