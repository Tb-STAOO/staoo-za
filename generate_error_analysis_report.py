#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parent
MODELS = ["cnn1d", "inceptiontime", "convnext1d"]
DEFAULT_DATA_ROOT = Path("F:/data/newdata")
OUTPUT_ROOT = PROJECT_DIR / "error_analysis"
TABLE_DIR = OUTPUT_ROOT / "tables"
FIG_DIR = OUTPUT_ROOT / "figures"
REPORT_PATH = PROJECT_DIR / "ERROR_ANALYSIS_REPORT.md"


def load_training_module():
    module_path = PROJECT_DIR / "train_binary_classifier.py"
    module_name = "train_binary_classifier"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def to_namespace_with_defaults(args_dict: Dict[str, Any]) -> SimpleNamespace:
    defaults = {
        "base_channels": 32,
        "kernel_size": 9,
        "dropout": 0.3,
        "inception_blocks": 6,
        "inception_channels": 32,
        "inception_bottleneck": 32,
        "convnext_dim1": 32,
        "convnext_dim2": 64,
        "convnext_dim3": 128,
        "convnext_depth1": 2,
        "convnext_depth2": 2,
        "convnext_depth3": 2,
        "convnext_kernel_size": 7,
        "convnext_drop_path": 0.1,
        "convnext_layer_scale": 1e-6,
        "convnext_use_grn": 1,
        "convnext_attention_pooling": 1,
    }
    merged = {**defaults, **(args_dict or {})}
    return SimpleNamespace(**merged)


def normalize_by_params(x: np.ndarray, norm_params: Dict[str, Any]) -> np.ndarray:
    mode = norm_params.get("mode")
    if mode is None:
        if "mean" in norm_params and "std" in norm_params:
            mode = "zscore"
        elif "min" in norm_params and "max" in norm_params:
            mode = "minmax"
        else:
            raise ValueError(f"Unsupported normalization params: {norm_params}")

    if mode == "zscore":
        mean = float(norm_params["mean"])
        std = float(norm_params["std"]) + 1e-8
        return (x - mean) / std
    if mode == "minmax":
        min_v = float(norm_params["min"])
        max_v = float(norm_params["max"])
        span = max(max_v - min_v, 1e-8)
        return (x - min_v) / span
    raise ValueError(f"Unsupported normalization mode: {mode}")


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save_misclassified_csv(model_name: str, rows: List[Dict[str, Any]]) -> Path:
    path = TABLE_DIR / f"{model_name}_misclassified.csv"
    fields = [
        "sample_index",
        "file_path",
        "true_label",
        "pred_label",
        "prob_yes",
        "wrong_confidence",
        "distance_to_threshold",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def plot_top_errors(
    model_name: str,
    x_raw: np.ndarray,
    mis_rows: List[Dict[str, Any]],
    max_plots: int = 12,
) -> List[Path]:
    saved = []
    top_rows = sorted(mis_rows, key=lambda r: r["wrong_confidence"], reverse=True)[:max_plots]
    for rank, row in enumerate(top_rows, start=1):
        idx = int(row["sample_index"])
        signal = x_raw[idx]
        fig, ax = plt.subplots(figsize=(10, 3.2), dpi=140)
        ax.plot(signal, linewidth=1.0)
        ax.set_title(
            f"{model_name} | rank#{rank} idx={idx} | true={row['true_label']} "
            f"pred={row['pred_label']} prob_yes={row['prob_yes']:.4f}"
        )
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        save_path = FIG_DIR / f"{model_name}_error_rank{rank:02d}_idx{idx:03d}.png"
        fig.savefig(save_path)
        plt.close(fig)
        saved.append(save_path)
    return saved


def infer_and_collect(module, model_name: str, test_data) -> Dict[str, Any]:
    ckpt_path = PROJECT_DIR / "model" / model_name / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    args = to_namespace_with_defaults(checkpoint.get("args", {}))
    model = module.build_model(model_name, args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm_params = checkpoint.get("normalization", {})
    x_norm = normalize_by_params(test_data.x, norm_params)
    x_tensor = torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(1)
    y_true = test_data.y.astype(np.int64)

    with torch.no_grad():
        logits = model(x_tensor).cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)

    label_map = {0: "no", 1: "yes"}
    mis_idx = np.where(preds != y_true)[0]
    mis_rows: List[Dict[str, Any]] = []
    for idx in mis_idx:
        pred = int(preds[idx])
        row = {
            "sample_index": int(idx),
            "file_path": str(test_data.files[idx]),
            "true_label": label_map[int(y_true[idx])],
            "pred_label": label_map[pred],
            "prob_yes": float(probs[idx]),
            "wrong_confidence": float(probs[idx] if pred == 1 else (1.0 - probs[idx])),
            "distance_to_threshold": float(abs(probs[idx] - 0.5)),
        }
        mis_rows.append(row)

    total = int(len(y_true))
    errors = int(len(mis_rows))
    no_to_yes = int(np.sum((y_true == 0) & (preds == 1)))
    yes_to_no = int(np.sum((y_true == 1) & (preds == 0)))

    return {
        "model_name": model_name,
        "checkpoint_path": str(ckpt_path),
        "total_samples": total,
        "error_samples": errors,
        "accuracy": float(np.mean(preds == y_true)),
        "error_rate": float(errors / total if total > 0 else 0.0),
        "no_to_yes": no_to_yes,
        "yes_to_no": yes_to_no,
        "avg_wrong_confidence": float(np.mean([r["wrong_confidence"] for r in mis_rows]) if mis_rows else 0.0),
        "mis_rows": mis_rows,
    }


def save_summary_csv(rows: List[Dict[str, Any]]) -> Path:
    path = TABLE_DIR / "error_summary.csv"
    fields = [
        "model_name",
        "accuracy",
        "error_rate",
        "error_samples",
        "total_samples",
        "no_to_yes",
        "yes_to_no",
        "avg_wrong_confidence",
        "checkpoint_path",
        "misclassified_csv",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model_name": row["model_name"],
                    "accuracy": row["accuracy"],
                    "error_rate": row["error_rate"],
                    "error_samples": row["error_samples"],
                    "total_samples": row["total_samples"],
                    "no_to_yes": row["no_to_yes"],
                    "yes_to_no": row["yes_to_no"],
                    "avg_wrong_confidence": row["avg_wrong_confidence"],
                    "checkpoint_path": row["checkpoint_path"],
                    "misclassified_csv": row["misclassified_csv"],
                }
            )
    return path


def write_report(
    data_root: Path,
    summary_rows: List[Dict[str, Any]],
    summary_csv_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Error Sample Analysis Report")
    lines.append("")
    lines.append("## 1. Objective")
    lines.append("- Analyze misclassified test samples for three models.")
    lines.append("- Provide structured outputs: summary table, per-model misclassified lists, and waveform figures.")
    lines.append("")
    lines.append("## 2. Data and Models")
    lines.append(f"- Test dataset root: `{data_root}`")
    lines.append(f"- Models: `{', '.join(MODELS)}`")
    lines.append("- Test set size: 200 samples (100 no + 100 yes).")
    lines.append("")
    lines.append("## 3. Error Summary")
    lines.append(f"- Summary CSV: `{summary_csv_path}`")
    lines.append("")
    lines.append("| Model | Accuracy | Error Rate | Errors | no->yes | yes->no | Avg Wrong Confidence |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(summary_rows, key=lambda r: r["accuracy"], reverse=True):
        lines.append(
            f"| {row['model_name']} | {row['accuracy']:.4f} | {row['error_rate']:.4f} | "
            f"{row['error_samples']}/{row['total_samples']} | {row['no_to_yes']} | "
            f"{row['yes_to_no']} | {row['avg_wrong_confidence']:.4f} |"
        )

    lines.append("")
    lines.append("## 4. Per-Model Error Analysis")
    for row in sorted(summary_rows, key=lambda r: r["accuracy"], reverse=True):
        lines.append(f"### 4.{summary_rows.index(row) + 1} {row['model_name']}")
        lines.append(f"- Checkpoint: `{row['checkpoint_path']}`")
        lines.append(f"- Misclassified list (CSV): `{row['misclassified_csv']}`")
        lines.append(f"- Error pattern: `no->yes={row['no_to_yes']}`, `yes->no={row['yes_to_no']}`")
        fig_preview = row["error_figures"][:6]
        if fig_preview:
            lines.append("- Typical error waveform plots:")
            for fig in fig_preview:
                lines.append(f"  - `{fig}`")
        else:
            lines.append("- No misclassified samples for this model.")
        lines.append("")

    best = max(summary_rows, key=lambda r: r["accuracy"])
    worst = min(summary_rows, key=lambda r: r["accuracy"])
    lines.append("## 5. Conclusion and Suggestions")
    lines.append(
        f"- Best model by accuracy: `{best['model_name']}` ({best['accuracy']:.4f})."
    )
    lines.append(
        f"- Highest error model: `{worst['model_name']}` ({worst['error_rate']:.4f} error rate)."
    )
    lines.append("- Suggested next steps:")
    lines.append("  - Add class-wise threshold tuning on validation set.")
    lines.append("  - Add hard-sample mining for high-confidence wrong predictions.")
    lines.append("  - Apply stronger regularization only when overfitting appears.")
    lines.append("")
    lines.append("## 6. Output Paths")
    lines.append(f"- Root folder: `{OUTPUT_ROOT}`")
    lines.append(f"- Report file: `{REPORT_PATH}`")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    module = load_training_module()

    data_root = DEFAULT_DATA_ROOT
    test_data = module.load_split(data_root, "test", expected_len=None)

    summary_rows: List[Dict[str, Any]] = []
    for model_name in MODELS:
        result = infer_and_collect(module, model_name, test_data)
        csv_path = save_misclassified_csv(model_name, result["mis_rows"])
        fig_paths = plot_top_errors(model_name, test_data.x, result["mis_rows"], max_plots=12)

        result["misclassified_csv"] = str(csv_path)
        result["error_figures"] = [str(p) for p in fig_paths]
        summary_rows.append(result)

    summary_csv_path = save_summary_csv(summary_rows)
    write_report(data_root, summary_rows, summary_csv_path)

    payload = {
        "data_root": str(data_root),
        "summary_csv": str(summary_csv_path),
        "report_markdown": str(REPORT_PATH),
        "models": [
            {
                "model_name": row["model_name"],
                "accuracy": row["accuracy"],
                "error_rate": row["error_rate"],
                "misclassified_csv": row["misclassified_csv"],
                "error_figures": row["error_figures"],
            }
            for row in summary_rows
        ],
    }
    (OUTPUT_ROOT / "error_analysis_index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Generated:")
    print(REPORT_PATH)
    print(summary_csv_path)
    print(OUTPUT_ROOT / "error_analysis_index.json")
    for row in summary_rows:
        print(f"{row['model_name']}: errors={row['error_samples']}/{row['total_samples']}, acc={row['accuracy']:.4f}")


if __name__ == "__main__":
    main()
