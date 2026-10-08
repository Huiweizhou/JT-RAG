import csv
import math
from pathlib import Path

from .anes_io import load_samples, prefer_zst_or_tsv, write_json
from .eval_metrics import compute_binary_metrics


def validation_target(processed_root, time_points, train_time):

    times = [int(t) for t in time_points]
    if times != sorted(set(times)):
        raise ValueError("TIME_POINTS must be strictly increasing.")
    index = times.index(int(train_time))
    if index + 1 >= len(times):
        raise ValueError(f"No next time point for training year {train_time}.")
    valid_time = times[index + 1]
    path = prefer_zst_or_tsv(Path(processed_root) / "random" / f"{valid_time}.valid.tsv")
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation split: {path}. Rerun data preprocessing.")
    return valid_time, path


def record_validation_summary(output_dir, metrics):

    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metrics_summary.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["validations"] if path.exists() else []
    rows = [row for row in rows if row["train_time"] != metrics["train_time"]]
    rows.append(metrics)
    rows.sort(key=lambda row: row["train_time"])
    write_json(path, {"validations": rows})
    fields = ["train_time", "valid_time", "context_time", "n_total", "accuracy",
              "precision", "recall", "f1", "roc_auc", "pr_auc", "csv"]
    with (output_dir / "metrics_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_next_validation(anes_model, qwen, cfg, train_time, output_csv):

    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    from .anes_dataset import ANESTimeDataset, anes_collate
    from .prompt_builder import build_prompt

    valid_time, split_file = validation_target(cfg["PROCESSED_ROOT"], cfg["TIME_POINTS"], train_time)
    dataset = ANESTimeDataset(
        processed_root=Path(cfg["PROCESSED_ROOT"]), time_points=cfg["TIME_POINTS"],
        sample_time=valid_time, candidate_max_size=int(cfg["CANDIDATE_MAX_SIZE"]),
        use_weak=False, sample_file=split_file,
    )
    if not dataset.samples:
        raise ValueError(f"Validation split is empty: {split_file}")
    sample_by_qid = {s.qid: s for s in dataset.samples}
    if len(sample_by_qid) != len(dataset.samples):
        raise ValueError(f"Duplicate qids in validation split: {split_file}")
    loader = DataLoader(dataset, batch_size=int(cfg["VALIDATION_BATCH_SIZE"]),
                        shuffle=False, num_workers=0, collate_fn=anes_collate)
    rows = []
    was_training = anes_model.training
    anes_model.eval()
    qwen.model.eval()
    try:
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Validation {train_time} -> {valid_time}",
                              disable=not cfg.get("SHOW_PROGRESS", True)):
                batch = {k: v.to(cfg["DEVICE"]) if torch.is_tensor(v) else v for k, v in batch.items()}
                selected = anes_model.select_evidence(batch, k_max=int(cfg["EVIDENCE_K_MAX"]), min_select=0)
                records, prompts = [], []
                for b, qid in enumerate(batch["qid"].tolist()):
                    sample = sample_by_qid[int(qid)]
                    count = int(selected["selected_counts"][b].item())
                    positions = selected["selected_candidate_positions"][b, :count].tolist()
                    evidence = [int(batch["candidate_ids"][b, p].item()) for p in positions]
                    prompts.append(build_prompt(sample, evidence, dataset.entities, dataset.graph))
                    records.append({"train_time": int(train_time), "valid_time": valid_time,
                                    "qid": sample.qid, "src": sample.src, "dst": sample.dst,
                                    "label": sample.label, "selected_count": count,
                                    "selected_eids": " ".join(map(str, evidence))})
                scores = qwen.score_yes_no_batch(prompts, score_batch_size=int(cfg["VALIDATION_SCORE_BATCH_SIZE"]))
                if len(scores) != len(records):
                    raise RuntimeError("Validation scorer returned an unexpected number of results.")
                for record, score in zip(records, scores):
                    pred = int(score.get("probability_pred", -1))
                    probability = float(score.get("p_yes", float("nan")))
                    if not score.get("score_valid") or pred not in (0, 1) or not math.isfinite(probability):
                        raise RuntimeError(f"Invalid validation score for qid={record['qid']}; metrics were not saved.")
                    record.update(pred=pred, p_yes=probability, score_valid=1, invalid=0)
                    rows.append(record)
    finally:
        anes_model.train(was_training)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = compute_binary_metrics(rows, score_key="p_yes")
    metrics.update(train_time=int(train_time), valid_time=valid_time,
                   context_time=int(dataset.context_time), split="valid", split_file=str(split_file),
                   primary_prediction="yes_no_probability_argmax", csv=str(output_csv))
    write_json(output_csv.with_suffix(".metrics.json"), metrics)
    return metrics


def check_validation_inputs(cfg):

    for train_time in cfg["JOINT_TRAIN_TIMES"]:
        valid_time, path = validation_target(cfg["PROCESSED_ROOT"], cfg["TIME_POINTS"], train_time)
        if not load_samples(Path(cfg["PROCESSED_ROOT"]), valid_time, sample_file=path):
            raise ValueError(f"Validation split is empty: {path}")
