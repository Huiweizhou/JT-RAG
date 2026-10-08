from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import yaml


DEFAULT_TIME_POINTS = [1990, 2000, 2005, 2010, 2014, 2017, 2019, 2020, 2021, 2022, 2023, 2024]
PairRaw = Tuple[str, str]
PairEid = Tuple[int, int]


def open_text(path: Path, mode: str = "rt", encoding: str = "utf-8"):

    path = Path(path)
    if path.suffix == ".zst":
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "读写 .zst 文件需要安装 zstandard：pip install zstandard"
            ) from exc
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


def write_single_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> int:

    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open_text(path, "wt") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)
            n += 1
    return n


def write_tsv_family(
    tsv_path: Path,
    header: Sequence[str],
    rows: Iterable[Sequence[object]],
    write_tsv: bool = True,
    write_zst: bool = True,
) -> int:

    rows_list = list(rows)
    if not write_tsv and not write_zst:
        raise ValueError("WRITE_TSV 和 WRITE_ZST 不能同时为 False")
    if write_tsv:
        write_single_tsv(tsv_path, header, rows_list)
    if write_zst:
        write_single_tsv(Path(str(tsv_path) + ".zst"), header, rows_list)
    return len(rows_list)


def existing_table_path(tsv_path: Path) -> Path:

    zst_path = Path(str(tsv_path) + ".zst")
    if zst_path.exists():
        return zst_path
    if tsv_path.exists():
        return tsv_path
    raise FileNotFoundError(f"找不到表文件：{zst_path} 或 {tsv_path}")


def read_tsv_table(tsv_path: Path) -> Iterator[Dict[str, str]]:
    path = existing_table_path(tsv_path)
    with open_text(path, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def stable_int_seed(text: str, base_seed: int = 0) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(base_seed)) & 0xFFFFFFFF


def canonical_pair_raw(a: str, b: str) -> PairRaw:
    a, b = str(a), str(b)
    return (a, b) if a <= b else (b, a)


def canonical_pair_eid(a: int, b: int) -> PairEid:
    a, b = int(a), int(b)
    return (a, b) if a <= b else (b, a)


def parse_year(x: object) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"\d{4}", s)
    if not m:
        return None
    year = int(m.group(0))
    if 1800 <= year <= 2100:
        return year
    return None


def parse_pmid_list(tail: str) -> List[str]:

    tail = tail.strip()
    if not tail:
        return []
    try:
        value = ast.literal_eval(tail)
        if isinstance(value, (list, tuple, set)):
            return [str(x) for x in value]
        return [str(value)]
    except Exception:
        return [x.strip("'\"") for x in re.findall(r"[A-Za-z0-9_.:-]+", tail)]


def parse_positive_line(line: str) -> Optional[Tuple[str, str, int, List[str]]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(maxsplit=3)
    if len(parts) < 3:
        return None
    raw_src, raw_dst = parts[0], parts[1]
    if raw_src.lower() in {"src", "source", "entity1"}:
        return None
    try:
        freq = int(float(parts[2]))
    except ValueError:
        return None
    pmids = parse_pmid_list(parts[3]) if len(parts) >= 4 else []
    if raw_src == raw_dst:
        return None
    return str(raw_src), str(raw_dst), freq, pmids


def parse_negative_line(line: str) -> Optional[Tuple[str, str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 2:
        return None
    raw_src, raw_dst = parts[0], parts[1]
    if raw_src.lower() in {"src", "source", "entity1"}:
        return None
    if raw_src == raw_dst:
        return None
    return str(raw_src), str(raw_dst)


@dataclass
class EntityInfo:
    raw_id: str
    name_counter: Counter = field(default_factory=Counter)
    texts: Set[str] = field(default_factory=set)

    def final_name(self) -> str:
        if self.name_counter:
            return self.name_counter.most_common(1)[0][0]
        return self.raw_id

    def final_desc(self, max_texts: int, seed: int) -> str:
        texts = sorted(t for t in self.texts if t)
        if not texts:
            return ""
        if len(texts) > max_texts:
            rng = random.Random(stable_int_seed(self.raw_id, seed))
            texts = rng.sample(texts, max_texts)
        return " ".join(t.strip() for t in texts if t.strip())


@dataclass
class RawEdgeRecord:
    raw_src: str
    raw_dst: str
    freq: int
    pmids: Set[str]
    snapshot_time: int

    @property
    def pair(self) -> PairRaw:
        return canonical_pair_raw(self.raw_src, self.raw_dst)


@dataclass
class ProcessedEdge:
    src: int
    dst: int
    freq: int
    first_time: int

    @property
    def pair(self) -> PairEid:
        return canonical_pair_eid(self.src, self.dst)


class GraphView:


    def __init__(self, edges: Iterable[ProcessedEdge]):
        self.adj: Dict[int, Dict[int, ProcessedEdge]] = defaultdict(dict)
        for e in edges:
            src, dst = canonical_pair_eid(e.src, e.dst)
            pe = ProcessedEdge(src, dst, e.freq, e.first_time)
            self.adj[src][dst] = pe
            self.adj[dst][src] = pe

    def neighbors(self, eid: int) -> Set[int]:
        return set(self.adj.get(int(eid), {}).keys())

    def edge(self, a: int, b: int) -> Optional[ProcessedEdge]:
        return self.adj.get(int(a), {}).get(int(b))

    def degree(self, eid: int) -> int:
        return len(self.adj.get(int(eid), {}))

    def candidates(self, src: int, dst: int) -> Set[int]:

        return (self.neighbors(src) | self.neighbors(dst)) - {int(src), int(dst)}


class JTRAGDatasetBuilder:
    def __init__(
        self,
        entity_json: Path,
        positive_dir: Path,
        negative_dir: Optional[Path],
        out_dir: Path,
        time_points: Sequence[int] = DEFAULT_TIME_POINTS,
        pos_pattern: str = "KWgraph_all_keys_{time}.edgelist",
        neg_pattern: str = "KWgraph_all_keys_{time}.edgelist",
        max_desc_texts: int = 5,
        desc_seed: int = 2026,
        k_init: int = 8,
        positive_snapshot_mode: str = "cumulative",
        skip_first_time_samples: bool = True,
        sample_shuffle_seed: int = 2026,
        write_tsv: bool = True,
        write_zst: bool = True,
        weak_only_for_new_link: bool = True,
        valid_pos_size: int = 5000,
        valid_neg_size: int = 5000,
        valid_seed: int = 2026,
    ) -> None:
        self.valid_pos_size = int(valid_pos_size)
        self.valid_neg_size = int(valid_neg_size)
        self.valid_seed = int(valid_seed)
        self.random_split_dir_name = "random"
        if min(self.valid_pos_size, self.valid_neg_size) < 0:
            raise ValueError("Validation split sizes must be nonnegative.")
        self.entity_json = Path(entity_json)
        self.positive_dir = Path(positive_dir)
        self.negative_dir = Path(negative_dir) if negative_dir else None
        self.out_dir = Path(out_dir)
        self.time_points = [int(t) for t in time_points]
        self.pos_pattern = pos_pattern
        self.neg_pattern = neg_pattern
        self.max_desc_texts = int(max_desc_texts)
        self.desc_seed = int(desc_seed)
        self.k_init = int(k_init)
        if positive_snapshot_mode not in {"cumulative", "delta"}:
            raise ValueError("positive_snapshot_mode 必须是 'cumulative' 或 'delta'")
        self.positive_snapshot_mode = positive_snapshot_mode
        self.skip_first_time_samples = bool(skip_first_time_samples)
        self.sample_shuffle_seed = int(sample_shuffle_seed)
        self.write_tsv = bool(write_tsv)
        self.write_zst = bool(write_zst)
        self.weak_only_for_new_link = bool(weak_only_for_new_link)

        self.entities: Dict[str, EntityInfo] = {}
        self.pmid_entities: Dict[str, Set[str]] = defaultdict(set)
        self.pmid_year: Dict[str, int] = {}
        self.pos_edges_by_time_raw: Dict[int, Dict[PairRaw, RawEdgeRecord]] = defaultdict(dict)
        self.neg_pairs_by_time_raw: Dict[int, Set[PairRaw]] = defaultdict(set)
        self.raw_to_eid: Dict[str, int] = {}
        self.eid_to_raw: Dict[int, str] = {}
        self.pair_first_time_raw: Dict[PairRaw, int] = {}
        self.pair_first_pmids_raw: Dict[PairRaw, Set[str]] = defaultdict(set)
        self.graph_edges_by_time: Dict[int, List[ProcessedEdge]] = {}
        self.sample_rows_by_time: Dict[int, List[Tuple[int, int, int, int]]] = {}


    def log(self, msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    @property
    def sample_time_points(self) -> List[int]:
        if self.skip_first_time_samples:
            return self.time_points[1:]
        return list(self.time_points)

    def previous_time(self, t: int) -> Optional[int]:
        idx = self.time_points.index(int(t))
        if idx == 0:
            return None
        return self.time_points[idx - 1]

    def context_time_for_sample(self, t: int) -> Optional[int]:

        return self.previous_time(t)

    def pos_path(self, time: int) -> Path:
        return self.positive_dir / self.pos_pattern.format(time=time, T=time)

    def neg_path(self, time: int) -> Optional[Path]:
        if self.negative_dir is None:
            return None
        return self.negative_dir / self.neg_pattern.format(time=time, T=time)

    def _ensure_entity(self, raw_id: str) -> EntityInfo:
        raw_id = str(raw_id)
        if raw_id not in self.entities:
            self.entities[raw_id] = EntityInfo(raw_id=raw_id)
        return self.entities[raw_id]

    def write_table(self, tsv_path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> int:
        return write_tsv_family(tsv_path, header, rows, self.write_tsv, self.write_zst)


    def load_entitylist(self) -> None:
        self.log(f"[1/9] 读取 entitylist.json: {self.entity_json}")
        if not self.entity_json.exists():
            raise FileNotFoundError(self.entity_json)


        try:
            import ijson
            with open(self.entity_json, "rb") as f:
                iterator = ijson.kvitems(f, "")
                self._consume_entity_iterator(iterator)
        except ImportError:
            with open(self.entity_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._consume_entity_iterator(data.items())

        self.log(f"      实体数 = {len(self.entities):,}，含实体标注的 PMID 数 = {len(self.pmid_entities):,}")

    def _consume_entity_iterator(self, iterator: Iterable[Tuple[str, list]]) -> None:
        for key, mentions in iterator:
            if not isinstance(mentions, list):
                continue
            for rec in mentions:
                if not isinstance(rec, dict):
                    continue
                raw_id = str(rec.get("entity_ID") or key)
                info = self._ensure_entity(raw_id)

                name = str(rec.get("entity_name") or raw_id).strip()
                if name:
                    info.name_counter[name] += 1

                text = str(rec.get("text") or "").strip()
                if text:
                    info.texts.add(text)

                pmid = rec.get("pmid")
                if pmid is not None:
                    pmid = str(pmid)
                    self.pmid_entities[pmid].add(raw_id)
                    year = parse_year(rec.get("time"))
                    if year is not None:
                        if pmid not in self.pmid_year or year < self.pmid_year[pmid]:
                            self.pmid_year[pmid] = year

    def scan_edgelists(self) -> None:
        self.log("[2/9] 扫描正例图谱和负例文件。")
        for t in self.time_points:
            p = self.pos_path(t)
            if not p.exists():
                raise FileNotFoundError(f"找不到 {t} 年正例 edgelist: {p}")
            n_edges = 0
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parsed = parse_positive_line(line)
                    if parsed is None:
                        continue
                    raw_src, raw_dst, freq, pmids = parsed
                    self._ensure_entity(raw_src)
                    self._ensure_entity(raw_dst)
                    pair = canonical_pair_raw(raw_src, raw_dst)
                    prev = self.pos_edges_by_time_raw[t].get(pair)
                    if prev is None:
                        self.pos_edges_by_time_raw[t][pair] = RawEdgeRecord(
                            raw_src=pair[0], raw_dst=pair[1], freq=freq,
                            pmids=set(pmids), snapshot_time=t,
                        )
                    else:

                        prev.freq += freq
                        prev.pmids.update(pmids)
                    n_edges += 1
            self.log(f"      {t}: 正例边读取 {n_edges:,} 行，去重后 {len(self.pos_edges_by_time_raw[t]):,} 条")

            neg_path = self.neg_path(t)
            if neg_path is not None and neg_path.exists():
                n_neg = 0
                with open(neg_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parsed = parse_negative_line(line)
                        if parsed is None:
                            continue
                        raw_src, raw_dst = parsed
                        self._ensure_entity(raw_src)
                        self._ensure_entity(raw_dst)
                        self.neg_pairs_by_time_raw[t].add(canonical_pair_raw(raw_src, raw_dst))
                        n_neg += 1
                self.log(f"      {t}: 负例对读取 {n_neg:,} 行，去重后 {len(self.neg_pairs_by_time_raw[t]):,} 条")
            else:
                self.log(f"      {t}: 未找到负例文件，将只写正例样本。")

    def make_entity_mapping(self) -> None:
        self.log("[3/9] 构建稳定的 raw_id -> eid 映射。")
        raw_ids = sorted(self.entities.keys())
        self.raw_to_eid = {raw_id: i for i, raw_id in enumerate(raw_ids)}
        self.eid_to_raw = {i: raw_id for raw_id, i in self.raw_to_eid.items()}
        self.log(f"      num_entities = {len(self.raw_to_eid):,}")

    def compute_pair_first_times(self) -> None:

        self.log("[4/9] 计算实体对首次出现年份和首次出现 PMID。")
        first_by_pair: Dict[PairRaw, int] = {}
        pmids_by_pair_year: Dict[PairRaw, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))
        earliest_snapshot: Dict[PairRaw, int] = {}
        pmids_by_pair_snapshot: Dict[PairRaw, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))

        for t in sorted(self.time_points):
            for pair, rec in self.pos_edges_by_time_raw[t].items():

                if pair not in earliest_snapshot or t < earliest_snapshot[pair]:
                    earliest_snapshot[pair] = t
                pmids_by_pair_snapshot[pair][t].update(rec.pmids)


                for pmid in rec.pmids:
                    year = self.pmid_year.get(str(pmid))
                    if year is None:
                        continue
                    if pair not in first_by_pair or year < first_by_pair[pair]:
                        first_by_pair[pair] = year
                    pmids_by_pair_year[pair][year].add(str(pmid))


        for pair, snap_t in earliest_snapshot.items():
            if pair not in first_by_pair:
                first_by_pair[pair] = snap_t

        self.pair_first_time_raw = first_by_pair


        for pair in first_by_pair:
            if pair in pmids_by_pair_year and pmids_by_pair_year[pair]:
                min_year = min(pmids_by_pair_year[pair])
                self.pair_first_pmids_raw[pair] = set(pmids_by_pair_year[pair][min_year])
            else:
                snap_t = earliest_snapshot.get(pair)
                if snap_t is not None:
                    self.pair_first_pmids_raw[pair] = set(pmids_by_pair_snapshot[pair].get(snap_t, set()))

        self.log(f"      有 first_time 的实体对数 = {len(self.pair_first_time_raw):,}")


    def write_entities(self) -> None:
        self.log("[5/9] 写出实体表 entities.tsv / entities.tsv.zst。")
        out_path = self.out_dir / "data" / "entities.tsv"

        def rows():
            for raw_id, eid in sorted(self.raw_to_eid.items(), key=lambda x: x[1]):
                info = self.entities.get(raw_id) or EntityInfo(raw_id=raw_id)
                yield [eid, raw_id, info.final_name(), info.final_desc(self.max_desc_texts, self.desc_seed)]

        n = self.write_table(out_path, ["eid", "raw_id", "name", "desc"], rows())
        self.log(f"      写出 {n:,} 行 -> {out_path}(.zst)")


    def build_graph_edges(self) -> None:
        self.log("[6/9] 构建并写出历史图快照。")
        cumulative: Dict[PairRaw, RawEdgeRecord] = {}
        for t in self.time_points:
            if self.positive_snapshot_mode == "delta":
                for pair, rec in self.pos_edges_by_time_raw[t].items():
                    if pair not in cumulative:
                        cumulative[pair] = RawEdgeRecord(pair[0], pair[1], rec.freq, set(rec.pmids), t)
                    else:
                        cumulative[pair].freq += rec.freq
                        cumulative[pair].pmids.update(rec.pmids)
                source_edges = cumulative
            else:

                source_edges = self.pos_edges_by_time_raw[t]

            processed: Dict[PairEid, ProcessedEdge] = {}
            for pair, rec in source_edges.items():
                src = self.raw_to_eid[pair[0]]
                dst = self.raw_to_eid[pair[1]]
                src, dst = canonical_pair_eid(src, dst)
                first_time = self.pair_first_time_raw.get(pair, t)
                processed[(src, dst)] = ProcessedEdge(src, dst, int(rec.freq), int(first_time))

            edge_list = [processed[k] for k in sorted(processed)]
            self.graph_edges_by_time[t] = edge_list
            out_path = self.out_dir / "data" / "graph" / f"{t}.edges.tsv"
            n = self.write_table(
                out_path,
                ["src", "dst", "freq", "first_time"],
                ([e.src, e.dst, e.freq, e.first_time] for e in edge_list),
            )
            self.log(f"      {t}: 写出 {n:,} 条边 -> {out_path}(.zst)")

    def write_samples(self) -> None:

        self.log("[7/9] 写出 sample/{T}.tsv，并混合打乱正负样本。")
        skipped = set(self.time_points) - set(self.sample_time_points)
        if skipped:
            self.log(f"      跳过这些时间点的 sample 构造: {sorted(skipped)}")

        for t in self.sample_time_points:
            pos_pairs = set(self.pos_edges_by_time_raw[t].keys())
            neg_pairs = set(self.neg_pairs_by_time_raw.get(t, set())) - pos_pairs

            pair_label_rows: List[Tuple[int, int, int]] = []
            for pair in sorted(pos_pairs):
                src, dst = canonical_pair_eid(self.raw_to_eid[pair[0]], self.raw_to_eid[pair[1]])
                pair_label_rows.append((src, dst, 1))
            for pair in sorted(neg_pairs):
                src, dst = canonical_pair_eid(self.raw_to_eid[pair[0]], self.raw_to_eid[pair[1]])
                pair_label_rows.append((src, dst, 0))


            rng = random.Random(self.sample_shuffle_seed + int(t))
            rng.shuffle(pair_label_rows)

            rows: List[Tuple[int, int, int, int]] = []
            for qid, (src, dst, label) in enumerate(pair_label_rows):
                rows.append((qid, src, dst, label))

            self.sample_rows_by_time[t] = rows
            out_path = self.out_dir / "sample" / f"{t}.tsv"
            n = self.write_table(out_path, ["qid", "src", "dst", "label"], rows)
            pos_n = sum(1 for r in rows if r[3] == 1)
            neg_n = sum(1 for r in rows if r[3] == 0)
            self.log(f"      {t}: 写出 {n:,} 个样本，正例 {pos_n:,}，负例 {neg_n:,} -> {out_path}(.zst)")


    def write_random_valid_test_splits(self) -> None:

        self.log("[8/9] 构建 random 验证集/测试集文件。")
        for t in self.sample_time_points:
            rows = list(self.sample_rows_by_time.get(t, []))
            pos_rows = [r for r in rows if int(r[3]) == 1]
            neg_rows = [r for r in rows if int(r[3]) == 0]

            rng = random.Random(self.valid_seed + int(t))
            valid_pos = rng.sample(pos_rows, min(self.valid_pos_size, len(pos_rows)))
            valid_neg = rng.sample(neg_rows, min(self.valid_neg_size, len(neg_rows)))
            valid_rows = valid_pos + valid_neg
            rng.shuffle(valid_rows)

            valid_qids = {int(r[0]) for r in valid_rows}
            test_rows = [r for r in rows if int(r[0]) not in valid_qids]
            rng.shuffle(test_rows)


            split_dir = self.out_dir / self.random_split_dir_name
            valid_path = split_dir / f"{t}.valid.tsv"
            test_path = split_dir / f"{t}.test.tsv"
            n_valid = self.write_table(valid_path, ["qid", "src", "dst", "label"], valid_rows)
            n_test = self.write_table(test_path, ["qid", "src", "dst", "label"], test_rows)
            self.log(
                f"      {t}: valid {n_valid:,} 条 "
                f"(pos={sum(1 for r in valid_rows if r[3] == 1):,}, neg={sum(1 for r in valid_rows if r[3] == 0):,}); "
                f"test {n_test:,} 条 -> {split_dir}"
            )

    def build_graph_view(self, t: int) -> GraphView:
        return GraphView(self.graph_edges_by_time.get(int(t), []))

    def edge_first_time_eid(self, a: int, b: int) -> Optional[int]:
        raw_a = self.eid_to_raw[int(a)]
        raw_b = self.eid_to_raw[int(b)]
        return self.pair_first_time_raw.get(canonical_pair_raw(raw_a, raw_b))

    def rank_weak_nodes(self, nodes: Set[int], graph: GraphView, src: int, dst: int) -> List[int]:

        def score(v: int):
            e_src = graph.edge(src, v)
            e_dst = graph.edge(dst, v)
            freq_src = e_src.freq if e_src else 0
            freq_dst = e_dst.freq if e_dst else 0
            ft_src = e_src.first_time if e_src else -1
            ft_dst = e_dst.first_time if e_dst else -1
            common = int(e_src is not None and e_dst is not None)
            recency = max(ft_src, ft_dst)
            deg = graph.degree(v)
            return (common, freq_src + freq_dst, recency, -math.log1p(deg), -v)

        return sorted(nodes, key=score, reverse=True)[: self.k_init]

    def build_weak_evidence(self) -> None:
        self.log("[9/9] 构建并写出 sample/{T}.weak.tsv。")
        graph_cache: Dict[int, GraphView] = {}

        for t in self.sample_time_points:
            ctx_t = self.context_time_for_sample(t)
            if ctx_t is None:

                graph = GraphView([])
                ctx_t_int = -10**9
            else:
                ctx_t_int = int(ctx_t)
                if ctx_t_int not in graph_cache:
                    graph_cache[ctx_t_int] = self.build_graph_view(ctx_t_int)
                graph = graph_cache[ctx_t_int]

            weak_rows: List[Tuple[int, int]] = []

            for qid, src, dst, label in self.sample_rows_by_time.get(t, []):
                if int(label) != 1:
                    continue


                candidates = graph.candidates(src, dst)
                if not candidates:
                    continue

                raw_src = self.eid_to_raw[int(src)]
                raw_dst = self.eid_to_raw[int(dst)]
                pair = canonical_pair_raw(raw_src, raw_dst)
                pair_first_time = self.pair_first_time_raw.get(pair)


                is_new_link_in_window = (
                    pair_first_time is not None
                    and ctx_t_int < int(pair_first_time) <= int(t)
                )
                if self.weak_only_for_new_link and not is_new_link_in_window:
                    continue


                r_doc: Set[int] = set()
                if pair_first_time is not None and ctx_t_int < int(pair_first_time) <= int(t):
                    for pmid in self.pair_first_pmids_raw.get(pair, set()):

                        pmid_y = self.pmid_year.get(str(pmid))
                        if pmid_y is not None and not (ctx_t_int < int(pmid_y) <= int(t)):
                            continue
                        for raw_v in self.pmid_entities.get(str(pmid), set()):
                            v = self.raw_to_eid.get(raw_v)
                            if v is not None and v in candidates and v not in {src, dst}:
                                r_doc.add(int(v))


                r_bridge: Set[int] = set()
                n_src = graph.neighbors(src)
                n_dst = graph.neighbors(dst)
                for v in candidates:
                    ft_v_dst = self.edge_first_time_eid(v, dst)
                    ft_v_src = self.edge_first_time_eid(v, src)
                    if v in n_src and ft_v_dst is not None and ctx_t_int < int(ft_v_dst) <= int(t):
                        r_bridge.add(v)
                    if v in n_dst and ft_v_src is not None and ctx_t_int < int(ft_v_src) <= int(t):
                        r_bridge.add(v)


                r_common = n_src & n_dst
                r_common = {v for v in r_common if v in candidates and v not in {src, dst}}


                chosen = self.rank_weak_nodes(r_doc | r_bridge | r_common, graph, src, dst)


                for v in chosen:
                    if v in candidates:
                        weak_rows.append((qid, v))

            out_path = self.out_dir / "sample" / f"{t}.weak.tsv"
            n = self.write_table(out_path, ["qid", "evidence"], weak_rows)
            self.log(f"      {t}: 写出 {n:,} 行弱监督证据 -> {out_path}(.zst)")

    def write_meta(self, semantic_enabled: bool) -> None:
        meta = {
            "time_points": self.time_points,
            "sample_time_points": self.sample_time_points,
            "context_policy": {
                "sample_T_uses_graph": "previous_time_point",
                "skip_first_time_samples": self.skip_first_time_samples,
                "note": "例如 sample/2000 使用 data/graph/1990.edges.tsv(.zst) 构造一跳邻居和弱监督证据。",
            },
            "storage": {
                "table_format": "tsv_and_tsv.zst" if self.write_tsv and self.write_zst else ("tsv.zst" if self.write_zst else "tsv"),
                "entity_id_type": "int32",
                "edge_type": "undirected",
                "src_dst_order": "src_less_than_dst",
                "read_priority": "zst_then_tsv",
            },
            "entity": {
                "file": "data/entities.tsv",
                "zst_file": "data/entities.tsv.zst",
                "columns": ["eid", "raw_id", "name", "desc"],
                "desc_mode": "global_static",
                "max_random_texts_per_entity": self.max_desc_texts,
                "desc_random_seed": self.desc_seed,
            },
            "semantic_embedding": {
                "enabled": bool(semantic_enabled),
                "file": "data/entity_sem_emb.f16.npy",
                "dtype": "float16",
                "row_index": "eid",
                "text_input": "name_plus_desc",
            },
            "graph": {
                "path_pattern": "data/graph/{time}.edges.tsv",
                "zst_path_pattern": "data/graph/{time}.edges.tsv.zst",
                "columns": ["src", "dst", "freq", "first_time"],
                "source": "positive_edgelist",
                "discard_pmid_list": True,
            },
            "sample": {
                "path_pattern": "sample/{time}.tsv",
                "zst_path_pattern": "sample/{time}.tsv.zst",
                "columns": ["qid", "src", "dst", "label"],
                "positive_label": 1,
                "negative_label": 0,
                "shuffle_seed": self.sample_shuffle_seed,
                "first_time_sample_skipped": self.skip_first_time_samples,
            },
            "weak_evidence": {
                "path_pattern": "sample/{time}.weak.tsv",
                "zst_path_pattern": "sample/{time}.weak.tsv.zst",
                "columns": ["qid", "evidence"],
                "used_for": "ANES_initialization",
                "test_time_usage": False,
                "store_intermediate_sources": False,
                "construction": {
                    "context_graph": "previous_time_point_graph",
                    "candidate_constraint": "evidence must be in one-hop union candidates from previous graph",
                    "sources": ["R_doc", "R_bridge", "R_common"],
                    "combination": "union_then_rank_and_cap",
                    "k_init": self.k_init,
                    "weak_only_for_new_link": self.weak_only_for_new_link,
                    "R_doc": "entities appearing in first co-occurrence PMIDs of query pair within (prev(T), T], intersected with previous-graph one-hop candidates",
                },
            },
            "candidate_generation": {
                "mode": "dynamic_from_graph_at_runtime",
                "definition": "C(src,dst,T) = (N_prev(T)(src) union N_prev(T)(dst)) - {src,dst}",
                "exclude_query_entities": True,
                "candidate_max_size": 256,
                "final_evidence_size": "determined_by_STOP",
            },
            "random_validation": {
                "dir": "random",
                "valid_path_pattern": "random/{time}.valid.tsv",
                "test_path_pattern": "random/{time}.test.tsv",
                "valid_pos_size": self.valid_pos_size,
                "valid_neg_size": self.valid_neg_size,
                "valid_seed": self.valid_seed,
                "training_file": "sample/{time}.tsv",
            },
            "preprocess_args": {
                "positive_snapshot_mode": self.positive_snapshot_mode,
                "positive_pattern": self.pos_pattern,
                "negative_pattern": self.neg_pattern,
            },
        }
        out_path = self.out_dir / "data" / "meta.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
        self.log(f"      写出 meta -> {out_path}")

    def run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "data" / "graph").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "sample").mkdir(parents=True, exist_ok=True)

        self.load_entitylist()
        self.scan_edgelists()
        self.make_entity_mapping()
        self.compute_pair_first_times()
        self.write_entities()
        self.build_graph_edges()
        self.write_samples()
        self.write_random_valid_test_splits()
        self.build_weak_evidence()
        self.write_meta(semantic_enabled=False)
        self.log("完成。")


def make_builder_from_config(cfg: dict) -> JTRAGDatasetBuilder:
    return JTRAGDatasetBuilder(
        entity_json=Path(cfg["ENTITY_JSON"]),
        positive_dir=Path(cfg["POSITIVE_DIR"]),
        negative_dir=Path(cfg["NEGATIVE_DIR"]) if cfg.get("NEGATIVE_DIR") else None,
        out_dir=Path(cfg["OUT_DIR"]),
        time_points=cfg.get("TIME_POINTS", DEFAULT_TIME_POINTS),
        pos_pattern=cfg.get("POS_PATTERN", "KWgraph_all_keys_{time}.edgelist"),
        neg_pattern=cfg.get("NEG_PATTERN", "KWgraph_all_keys_{time}.edgelist"),
        max_desc_texts=int(cfg.get("MAX_DESC_TEXTS", 5)),
        desc_seed=int(cfg.get("DESC_RANDOM_SEED", 2026)),
        k_init=int(cfg.get("K_INIT", 8)),
        positive_snapshot_mode=cfg.get("POSITIVE_SNAPSHOT_MODE", "cumulative"),
        skip_first_time_samples=bool(cfg.get("SKIP_FIRST_TIME_SAMPLES", True)),
        sample_shuffle_seed=int(cfg.get("SAMPLE_SHUFFLE_SEED", 2026)),
        valid_pos_size=int(cfg.get("VALID_POS_SIZE", 5000)),
        valid_neg_size=int(cfg.get("VALID_NEG_SIZE", 5000)),
        valid_seed=int(cfg.get("VALID_RANDOM_SEED", 2026)),
        write_tsv=bool(cfg.get("WRITE_TSV", True)),
        write_zst=bool(cfg.get("WRITE_ZST", True)),
        weak_only_for_new_link=bool(cfg.get("WEAK_ONLY_FOR_NEW_LINK", True)),
    )
