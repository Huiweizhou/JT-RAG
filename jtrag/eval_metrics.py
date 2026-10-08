from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if float(b) != 0.0 else 0.0


def _rankdata_average(values: np.ndarray) -> np.ndarray:

    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _rankdata_average(y_score.astype(np.float64))
    sum_pos_ranks = float(ranks[pos].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = 0
    precision_sum = 0.0
    for i, y in enumerate(y_sorted, start=1):
        if int(y) == 1:
            tp += 1
            precision_sum += tp / float(i)
    return float(precision_sum / n_pos)


def compute_binary_metrics(rows: List[dict], score_key: str = "p_yes") -> Dict[str, Optional[float]]:

    n_total = len(rows)
    invalid_rows = [r for r in rows if int(r.get("invalid", 0)) != 0 or int(r.get("pred", -1)) < 0]
    valid_rows = [r for r in rows if int(r.get("invalid", 0)) == 0 and int(r.get("pred", -1)) in (0, 1)]

    y_true = np.asarray([int(r["label"]) for r in valid_rows], dtype=np.int64)
    y_pred = np.asarray([int(r["pred"]) for r in valid_rows], dtype=np.int64)

    n_valid = len(valid_rows)
    tp = int(((y_true == 1) & (y_pred == 1)).sum()) if n_valid else 0
    tn = int(((y_true == 0) & (y_pred == 0)).sum()) if n_valid else 0
    fp = int(((y_true == 0) & (y_pred == 1)).sum()) if n_valid else 0
    fn = int(((y_true == 1) & (y_pred == 0)).sum()) if n_valid else 0

    acc = _safe_div(tp + tn, n_valid)
    acc_with_invalid = _safe_div(tp + tn, n_total)
    pos_precision = _safe_div(tp, tp + fp)
    pos_recall = _safe_div(tp, tp + fn)
    pos_f1 = _safe_div(2 * pos_precision * pos_recall, pos_precision + pos_recall)
    neg_precision = _safe_div(tn, tn + fn)
    neg_recall = _safe_div(tn, tn + fp)
    neg_f1 = _safe_div(2 * neg_precision * neg_recall, neg_precision + neg_recall)
    macro_precision = (pos_precision + neg_precision) / 2.0
    macro_recall = (pos_recall + neg_recall) / 2.0
    macro_f1 = (pos_f1 + neg_f1) / 2.0
    support_pos = tp + fn
    support_neg = tn + fp
    weighted_f1 = _safe_div(pos_f1 * support_pos + neg_f1 * support_neg, support_pos + support_neg)


    micro_precision = acc
    micro_recall = acc
    micro_f1 = acc
    balanced_accuracy = (pos_recall + neg_recall) / 2.0
    specificity = neg_recall
    npv = neg_precision

    mcc_den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = _safe_div(tp * tn - fp * fn, mcc_den)
    expected_acc = 0.0
    if n_valid:
        pred_pos = tp + fp
        pred_neg = tn + fn
        true_pos = tp + fn
        true_neg = tn + fp
        expected_acc = ((pred_pos * true_pos) + (pred_neg * true_neg)) / float(n_valid * n_valid)
    kappa = _safe_div(acc - expected_acc, 1.0 - expected_acc)

    metrics: Dict[str, Optional[float]] = {
        "n_total": float(n_total),
        "n_valid": float(n_valid),
        "n_invalid": float(len(invalid_rows)),
        "invalid_ratio": _safe_div(len(invalid_rows), n_total),
        "accuracy": acc,
        "accuracy_with_invalid_as_wrong": acc_with_invalid,
        "balanced_accuracy": balanced_accuracy,
        "precision": pos_precision,
        "recall": pos_recall,
        "f1": pos_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "pos_precision": pos_precision,
        "pos_recall": pos_recall,
        "pos_f1": pos_f1,
        "neg_precision": neg_precision,
        "neg_recall": neg_recall,
        "neg_f1": neg_f1,
        "specificity": specificity,
        "npv": npv,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "avg_selected_count": float(np.mean([int(r.get("selected_count", 0)) for r in rows])) if rows else 0.0,
        "zero_evidence_ratio": float(np.mean([int(r.get("selected_count", 0)) == 0 for r in rows])) if rows else 0.0,
    }


    score_rows = [
        r for r in rows
        if int(r.get("score_valid", 1)) == 1 and r.get(score_key, "") not in (None, "")
    ]
    if score_rows:
        yt = np.asarray([int(r["label"]) for r in score_rows], dtype=np.int64)
        ys = np.asarray([float(r[score_key]) for r in score_rows], dtype=np.float64)
        ys = np.clip(ys, 1e-7, 1.0 - 1e-7)
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score

            metrics["roc_auc"] = float(roc_auc_score(yt, ys)) if len(set(yt.tolist())) == 2 else None
            metrics["pr_auc"] = float(average_precision_score(yt, ys)) if int((yt == 1).sum()) > 0 else None
        except Exception:
            metrics["roc_auc"] = roc_auc_binary(yt, ys)
            metrics["pr_auc"] = average_precision_binary(yt, ys)
        metrics["brier_score"] = float(np.mean((ys - yt.astype(np.float64)) ** 2))
        metrics["log_loss"] = float(-np.mean(yt * np.log(ys) + (1 - yt) * np.log(1 - ys)))
        metrics["mean_p_yes_pos"] = float(np.mean(ys[yt == 1])) if int((yt == 1).sum()) else None
        metrics["mean_p_yes_neg"] = float(np.mean(ys[yt == 0])) if int((yt == 0).sum()) else None
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["brier_score"] = None
        metrics["log_loss"] = None
        metrics["mean_p_yes_pos"] = None
        metrics["mean_p_yes_neg"] = None

    return metrics
