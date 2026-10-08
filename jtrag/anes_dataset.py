from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set

import numpy as np
import torch
from torch.utils.data import Dataset

from .anes_io import (
    GraphSnapshot,
    SampleRecord,
    load_entities,
    load_graph,
    load_samples,
    load_semantic_embeddings,
    load_struct_embeddings,
    load_weak_map,
    previous_time,
)


@dataclass
class CandidateInfo:


    candidate_id: int
    connect_to_src: int
    connect_to_dst: int
    freq_src: int
    freq_dst: int
    first_src: int
    first_dst: int
    degree: int
    is_common: int
    side_mask: int
    prefilter_score: float


class ANESFeatureBuilder:


    def __init__(
        self,
        graph: GraphSnapshot,
        context_time: int,
        sample_time: int,
        candidate_max_size: int = 256,
    ) -> None:
        self.graph = graph
        self.context_time = int(context_time)
        self.sample_time = int(sample_time)
        self.candidate_max_size = int(candidate_max_size)

    def build_candidates(self, sample: SampleRecord, weak_set: Optional[Set[int]] = None) -> List[CandidateInfo]:

        _ = weak_set
        src, dst = int(sample.src), int(sample.dst)
        candidates = self.graph.candidate_set(src, dst)

        infos: List[CandidateInfo] = []
        for v in candidates:
            e_src = self.graph.edge(src, v)
            e_dst = self.graph.edge(dst, v)
            connect_src = int(e_src is not None)
            connect_dst = int(e_dst is not None)
            freq_src = int(e_src.freq) if e_src else 0
            freq_dst = int(e_dst.freq) if e_dst else 0
            first_src = int(e_src.first_time) if e_src else -1
            first_dst = int(e_dst.first_time) if e_dst else -1
            degree = int(self.graph.degree(v))
            is_common = int(connect_src and connect_dst)
            side_mask = connect_src + 2 * connect_dst


            total_freq = freq_src + freq_dst
            recent = max(first_src, first_dst)
            prefilter_score = (
                is_common * 1e6
                + math.log1p(total_freq) * 1e4
                + recent
                - math.log1p(degree) * 10.0
            )
            infos.append(
                CandidateInfo(
                    candidate_id=int(v),
                    connect_to_src=connect_src,
                    connect_to_dst=connect_dst,
                    freq_src=freq_src,
                    freq_dst=freq_dst,
                    first_src=first_src,
                    first_dst=first_dst,
                    degree=degree,
                    is_common=is_common,
                    side_mask=side_mask,
                    prefilter_score=float(prefilter_score),
                )
            )

        infos.sort(key=lambda x: (x.prefilter_score, -x.candidate_id), reverse=True)
        return infos[: self.candidate_max_size]

    def build_tmp_and_meta(self, sample: SampleRecord, infos: List[CandidateInfo]) -> tuple[np.ndarray, np.ndarray]:

        src, dst = int(sample.src), int(sample.dst)
        src_degree = self.graph.degree(src)
        dst_degree = self.graph.degree(dst)
        max_time_span = max(1.0, float(self.sample_time - 1944 + 1))

        tmp_feats: List[List[float]] = []
        meta_feats: List[List[float]] = []

        for c in infos:

            src_known = float(c.first_src >= 0)
            dst_known = float(c.first_dst >= 0)
            delta_src = float(self.sample_time - c.first_src) if c.first_src >= 0 else 0.0
            delta_dst = float(self.sample_time - c.first_dst) if c.first_dst >= 0 else 0.0
            rec_src = 1.0 / (delta_src + 1.0) if src_known else 0.0
            rec_dst = 1.0 / (delta_dst + 1.0) if dst_known else 0.0
            latest_first = max(c.first_src, c.first_dst)
            latest_delta = float(self.sample_time - latest_first) if latest_first >= 0 else 0.0
            latest_rec = 1.0 / (latest_delta + 1.0) if latest_first >= 0 else 0.0


            tmp_feats.append(
                [
                    delta_src / max_time_span,
                    delta_dst / max_time_span,
                    rec_src,
                    rec_dst,
                    math.log1p(c.freq_src),
                    math.log1p(c.freq_dst),
                    src_known,
                    dst_known,
                    latest_delta / max_time_span,
                    latest_rec,
                ]
            )

            total_freq = c.freq_src + c.freq_dst
            freq_diff = abs(c.freq_src - c.freq_dst)
            freq_balance = 1.0 - float(freq_diff) / float(total_freq + 1)

            meta_feats.append(
                [
                    float(c.connect_to_src),
                    float(c.connect_to_dst),
                    float(c.is_common),
                    float(c.side_mask == 1),
                    float(c.side_mask == 2),
                    float(c.side_mask == 3),
                    math.log1p(c.degree),
                    1.0 / math.log(c.degree + 2.0),
                    math.log1p(total_freq),
                    freq_balance,
                    math.log1p(src_degree),
                    math.log1p(dst_degree),
                    float(c.degree > 0),
                ]
            )

        if len(infos) == 0:
            return np.zeros((0, 10), dtype=np.float32), np.zeros((0, 13), dtype=np.float32)
        return np.asarray(tmp_feats, dtype=np.float32), np.asarray(meta_feats, dtype=np.float32)


class ANESTimeDataset(Dataset):


    def __init__(
        self,
        processed_root: Path,
        time_points: Sequence[int],
        sample_time: int,
        candidate_max_size: int = 256,
        qid_filter: Optional[Set[int]] = None,
        use_weak: bool = True,
        mmap_mode: str = "r",
        sample_file: Optional[Path] = None,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.time_points = [int(t) for t in time_points]
        self.sample_time = int(sample_time)
        self.context_time = previous_time(self.time_points, self.sample_time)
        self.candidate_max_size = int(candidate_max_size)

        self.entities = load_entities(self.processed_root)
        self.samples = load_samples(self.processed_root, self.sample_time, sample_file=sample_file)
        if qid_filter is not None:
            qid_filter = {int(x) for x in qid_filter}
            self.samples = [s for s in self.samples if s.qid in qid_filter]

        self.weak_map = load_weak_map(self.processed_root, self.sample_time) if use_weak else {}
        self.graph = load_graph(self.processed_root, self.context_time)
        self.sem = load_semantic_embeddings(self.processed_root, mmap_mode=mmap_mode)
        self.struct = load_struct_embeddings(self.processed_root, self.context_time, mmap_mode=mmap_mode)
        self.builder = ANESFeatureBuilder(
            graph=self.graph,
            context_time=self.context_time,
            sample_time=self.sample_time,
            candidate_max_size=self.candidate_max_size,
        )


        n_entities = len(self.entities)
        if self.sem.shape[0] != n_entities:
            raise ValueError(f"语义向量行数 {self.sem.shape[0]} 与实体数 {n_entities} 不一致。")
        if self.struct.shape[0] != n_entities:
            raise ValueError(f"结构向量行数 {self.struct.shape[0]} 与实体数 {n_entities} 不一致。")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        weak_set = self.weak_map.get(sample.qid, set())
        full_candidate_set = self.graph.candidate_set(int(sample.src), int(sample.dst))
        weak_total_in_full_candidate = len(set(weak_set) & full_candidate_set)
        infos = self.builder.build_candidates(sample, weak_set=weak_set)
        candidate_ids = np.asarray([c.candidate_id for c in infos], dtype=np.int64)
        tmp_feats, meta_feats = self.builder.build_tmp_and_meta(sample, infos)


        weak_in_candidate = set(int(x) for x in candidate_ids.tolist()) & set(weak_set)
        weak_labels = np.asarray([1.0 if int(cid) in weak_in_candidate else 0.0 for cid in candidate_ids], dtype=np.float32)

        if len(candidate_ids) > 0:
            cand_sem = np.asarray(self.sem[candidate_ids], dtype=np.float32)
            cand_str = np.asarray(self.struct[candidate_ids], dtype=np.float32)
        else:
            cand_sem = np.zeros((0, self.sem.shape[1]), dtype=np.float32)
            cand_str = np.zeros((0, self.struct.shape[1]), dtype=np.float32)

        return {
            "qid": int(sample.qid),
            "src": int(sample.src),
            "dst": int(sample.dst),
            "label": int(sample.label),
            "sample_time": int(self.sample_time),
            "context_time": int(self.context_time),
            "candidate_ids": candidate_ids,
            "candidate_sem": cand_sem,
            "candidate_str": cand_str,
            "candidate_tmp": tmp_feats,
            "candidate_meta": meta_feats,
            "candidate_weak": weak_labels,
            "weak_total_in_full_candidate": int(weak_total_in_full_candidate),
            "weak_retained_count": int(len(weak_in_candidate)),
            "src_sem": np.asarray(self.sem[sample.src], dtype=np.float32),
            "dst_sem": np.asarray(self.sem[sample.dst], dtype=np.float32),
            "src_str": np.asarray(self.struct[sample.src], dtype=np.float32),
            "dst_str": np.asarray(self.struct[sample.dst], dtype=np.float32),
        }


def anes_collate(batch: Sequence[dict]) -> dict:

    bsz = len(batch)
    max_c = max((len(x["candidate_ids"]) for x in batch), default=0)

    sem_dim = int(batch[0]["src_sem"].shape[0])
    str_dim = int(batch[0]["src_str"].shape[0])
    tmp_dim = int(batch[0]["candidate_tmp"].shape[1]) if max_c > 0 else 10
    meta_dim = int(batch[0]["candidate_meta"].shape[1]) if max_c > 0 else 13

    candidate_ids = torch.full((bsz, max_c), -1, dtype=torch.long)
    candidate_mask = torch.zeros((bsz, max_c), dtype=torch.bool)
    candidate_sem = torch.zeros((bsz, max_c, sem_dim), dtype=torch.float32)
    candidate_str = torch.zeros((bsz, max_c, str_dim), dtype=torch.float32)
    candidate_tmp = torch.zeros((bsz, max_c, tmp_dim), dtype=torch.float32)
    candidate_meta = torch.zeros((bsz, max_c, meta_dim), dtype=torch.float32)
    candidate_weak = torch.zeros((bsz, max_c), dtype=torch.float32)

    for i, item in enumerate(batch):
        n = len(item["candidate_ids"])
        if n == 0:
            continue
        candidate_ids[i, :n] = torch.as_tensor(item["candidate_ids"], dtype=torch.long)
        candidate_mask[i, :n] = True
        candidate_sem[i, :n] = torch.as_tensor(item["candidate_sem"], dtype=torch.float32)
        candidate_str[i, :n] = torch.as_tensor(item["candidate_str"], dtype=torch.float32)
        candidate_tmp[i, :n] = torch.as_tensor(item["candidate_tmp"], dtype=torch.float32)
        candidate_meta[i, :n] = torch.as_tensor(item["candidate_meta"], dtype=torch.float32)
        candidate_weak[i, :n] = torch.as_tensor(item["candidate_weak"], dtype=torch.float32)

    return {
        "qid": torch.as_tensor([x["qid"] for x in batch], dtype=torch.long),
        "src": torch.as_tensor([x["src"] for x in batch], dtype=torch.long),
        "dst": torch.as_tensor([x["dst"] for x in batch], dtype=torch.long),
        "label": torch.as_tensor([x["label"] for x in batch], dtype=torch.long),
        "sample_time": torch.as_tensor([x["sample_time"] for x in batch], dtype=torch.long),
        "context_time": torch.as_tensor([x["context_time"] for x in batch], dtype=torch.long),
        "candidate_ids": candidate_ids,
        "candidate_mask": candidate_mask,
        "candidate_sem": candidate_sem,
        "candidate_str": candidate_str,
        "candidate_tmp": candidate_tmp,
        "candidate_meta": candidate_meta,
        "candidate_weak": candidate_weak,
        "weak_total_in_full_candidate": torch.as_tensor(
            [x["weak_total_in_full_candidate"] for x in batch], dtype=torch.long
        ),
        "weak_retained_count": torch.as_tensor([x["weak_retained_count"] for x in batch], dtype=torch.long),
        "src_sem": torch.as_tensor(np.stack([x["src_sem"] for x in batch]), dtype=torch.float32),
        "dst_sem": torch.as_tensor(np.stack([x["dst_sem"] for x in batch]), dtype=torch.float32),
        "src_str": torch.as_tensor(np.stack([x["src_str"] for x in batch]), dtype=torch.float32),
        "dst_str": torch.as_tensor(np.stack([x["dst_str"] for x in batch]), dtype=torch.float32),
    }


def balanced_qid_sample(
    processed_root: Path,
    sample_time: int,
    pos_n: int = 500,
    neg_n: int = 500,
    seed: int = 2026,
) -> Set[int]:

    samples = load_samples(Path(processed_root), int(sample_time))
    pos = [s.qid for s in samples if int(s.label) == 1]
    neg = [s.qid for s in samples if int(s.label) == 0]
    rng = random.Random(int(seed) + int(sample_time))
    rng.shuffle(pos)
    rng.shuffle(neg)
    qids = pos[: min(pos_n, len(pos))] + neg[: min(neg_n, len(neg))]
    rng.shuffle(qids)
    return set(int(x) for x in qids)
