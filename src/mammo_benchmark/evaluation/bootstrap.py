"""Bootstrap confidence intervals from prediction CSVs saved by foundation_model.

Reads predictions_<dataset>_<task>.csv files written to <run>/eval/ by
FoundationLightningModule.on_test_epoch_end. Prediction CSVs may include
metadata columns such as exam_id and num_images; metrics are computed from
prob_0,...,prob_C-1 plus label.

Usage:
    python -m utils.bootstrap_metrics output-foundation/<run_name>/eval/
    python -m utils.bootstrap_metrics <eval_dir> --n_boot 1000 --ci 95 --output bootstrap.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Match foundation_model metrics.

    For binary tasks, ``auprc`` is the positive-class average precision
    computed from prob_1, not a macro average over both classes.
    """
    num_classes = probs.shape[1]
    out: dict = {}
    onehot = np.eye(num_classes)[labels]

    try:
        if num_classes == 2:
            out["auc"] = float(roc_auc_score(labels, probs[:, 1]))
            out["auprc"] = float(average_precision_score(labels, probs[:, 1]))
        else:
            out["auc"] = float(roc_auc_score(onehot, probs, multi_class="ovr", average="macro"))
            weights = np.arange(num_classes)
            expected = (probs * weights).sum(axis=1)
            corr, _ = spearmanr(expected, labels)
            if np.isfinite(corr):
                out["spearman"] = float(corr)
    except ValueError:
        # Bootstrap resample may have only one class → roc_auc_score raises.
        pass
    return out


def bootstrap(probs: np.ndarray, labels: np.ndarray, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    samples: dict = {}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for metric, value in compute_metrics(probs[idx], labels[idx]).items():
            samples.setdefault(metric, []).append(value)
    return {k: np.asarray(v, dtype=np.float64) for k, v in samples.items()}


def summarize(point: dict, boot: dict, ci: float) -> dict:
    alpha = (100.0 - ci) / 2.0
    summary = {}
    for metric, value in point.items():
        arr = boot.get(metric, np.array([]))
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            lo = float(np.percentile(arr, alpha))
            hi = float(np.percentile(arr, 100 - alpha))
            std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
        else:
            lo = hi = std = float("nan")
        summary[metric] = {
            "pretty": f"{float(value):.4f} ± {std:.4f} [{lo:.4f}, {hi:.4f}]",
            "value": float(value),
            "ci_low": lo,
            "ci_high": hi,
            "std": std,
            "n_boot_valid": int(arr.size),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CIs from prediction CSVs.")
    parser.add_argument("eval_dir", type=str, help="Directory with predictions_<dataset>_<task>.csv files.")
    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ci", type=float, default=95.0, help="Confidence interval percent (e.g. 95).")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    csv_files = sorted(eval_dir.glob("predictions_*.csv"))
    if not csv_files:
        raise SystemExit(f"No predictions_*.csv found in {eval_dir}")

    results = {}
    for path in csv_files:
        df = pd.read_csv(path)
        prob_cols = [c for c in df.columns if c.startswith("prob_")]
        probs = df[prob_cols].to_numpy(dtype=np.float64)
        labels = df["label"].to_numpy(dtype=np.int64)
        if labels.size == 0:
            continue

        point = compute_metrics(probs, labels)
        boot = bootstrap(probs, labels, n_boot=args.n_boot, seed=args.seed)
        summary = summarize(point, boot, ci=args.ci)
        summary["n_samples"] = int(labels.size)
        results[path.stem] = summary

        tag = path.stem.replace("predictions_", "")
        parts = [f"{m}={s['value']:.4f} [{s['ci_low']:.4f}, {s['ci_high']:.4f}]" for m, s in summary.items() if m != "n_samples"]
        print(f"[{tag}] N={summary['n_samples']}  " + "  ".join(parts))

    if args.output:
        payload = {
            "config": {"n_boot": args.n_boot, "seed": args.seed, "ci": args.ci, "eval_dir": str(eval_dir)},
            "results": results,
        }
        Path(args.output).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved bootstrap summary to {args.output}")


if __name__ == "__main__":
    main()
