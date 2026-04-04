#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODELS = ["cnn1d", "inceptiontime", "convnext1d"]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_best_attempt(payload: dict) -> dict:
    best_seed = payload.get("best_seed")
    attempts = payload.get("all_attempts", [])
    for item in attempts:
        if item.get("seed") == best_seed:
            return item
    if attempts:
        return max(attempts, key=lambda x: x.get("metrics", {}).get("test_acc", -1))
    raise RuntimeError("No attempts found in metrics payload")


def plot_training_curve(history: list[dict], model_name: str, save_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_acc = [h["train_acc"] for h in history]
    val_acc = [h["val_acc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    ax1, ax2 = axes

    ax1.plot(epochs, train_loss, label="train_loss", linewidth=1.8)
    ax1.plot(epochs, val_loss, label="val_loss", linewidth=1.8)
    ax1.set_title(f"{model_name} - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.25)
    ax1.legend()

    ax2.plot(epochs, train_acc, label="train_acc", linewidth=1.8)
    ax2.plot(epochs, val_acc, label="val_acc", linewidth=1.8)
    ax2.set_title(f"{model_name} - Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], model_name: str, save_path: Path) -> None:
    arr = np.array(cm, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=140)
    im = ax.imshow(arr, cmap="Blues")

    labels = ["no", "yes"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} - Confusion Matrix")

    max_val = arr.max() if arr.size else 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            color = "white" if arr[i, j] > max_val * 0.5 else "black"
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parent
    out_root = project_root / "output"
    report_root = out_root / "report"
    fig_root = report_root / "figures"

    report_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    rows = []
    data_root = None

    for model in MODELS:
        metrics_path = out_root / model / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

        payload = load_json(metrics_path)
        best_metrics = payload["best_metrics"]
        best_attempt = find_best_attempt(payload)
        history = best_attempt.get("history", [])
        cls_report = best_metrics.get("classification_report", {})

        if data_root is None:
            data_root = payload.get("data_root")

        curve_path = fig_root / f"{model}_training_curve.png"
        cm_path = fig_root / f"{model}_confusion_matrix.png"

        if history:
            plot_training_curve(history, model, curve_path)
        plot_confusion_matrix(best_metrics.get("confusion_matrix", [[0, 0], [0, 0]]), model, cm_path)

        row = {
            "model": model,
            "best_seed": payload.get("best_seed"),
            "best_val_acc": best_metrics.get("best_val_acc"),
            "test_acc": best_metrics.get("test_acc"),
            "test_precision": best_metrics.get("test_precision"),
            "test_recall": best_metrics.get("test_recall"),
            "test_f1": best_metrics.get("test_f1"),
            "test_roc_auc": best_metrics.get("test_roc_auc"),
            "confusion_matrix": best_metrics.get("confusion_matrix"),
            "no_precision": cls_report.get("no", {}).get("precision"),
            "no_recall": cls_report.get("no", {}).get("recall"),
            "yes_precision": cls_report.get("yes", {}).get("precision"),
            "yes_recall": cls_report.get("yes", {}).get("recall"),
            "epochs_used": len(history),
            "metrics_json": str(metrics_path),
            "curve_png": str(curve_path),
            "cm_png": str(cm_path),
        }
        rows.append(row)

    rows.sort(key=lambda x: x["test_acc"], reverse=True)

    csv_path = report_root / "metrics_summary.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md_path = report_root / "assignment_report.md"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# 一维波形二分类实验报告（自动生成）")
    lines.append("")
    lines.append(f"- 生成时间: {now}")
    lines.append(f"- 数据集路径: `{data_root}`")
    lines.append(f"- 模型数量: {len(rows)}（cnn1d / inceptiontime / convnext1d）")
    lines.append("")
    lines.append("## 1. 评价指标对比")
    lines.append("")
    lines.append("| Rank | Model | Test Acc | Precision | Recall | F1 | ROC-AUC | Best Seed |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {model} | {acc:.4f} | {pre:.4f} | {rec:.4f} | {f1:.4f} | {auc:.4f} | {seed} |".format(
                rank=i,
                model=r["model"],
                acc=r["test_acc"],
                pre=r["test_precision"],
                rec=r["test_recall"],
                f1=r["test_f1"],
                auc=r["test_roc_auc"],
                seed=r["best_seed"],
            )
        )

    best = rows[0]
    lines.append("")
    lines.append("## 2. 最优模型")
    lines.append("")
    lines.append(
        "- 最优模型: **{model}**，Test Acc={acc:.4f}，F1={f1:.4f}，ROC-AUC={auc:.4f}".format(
            model=best["model"],
            acc=best["test_acc"],
            f1=best["test_f1"],
            auc=best["test_roc_auc"],
        )
    )
    lines.append(f"- 混淆矩阵: `{best['confusion_matrix']}`")

    lines.append("")
    lines.append("## 3. 作业要求输出文件")
    lines.append("")
    lines.append(f"- 指标总表: `{csv_path}`")
    lines.append(f"- 模型对比JSON: `{out_root / 'model_comparison.json'}`")
    lines.append(f"- 模型对比CSV: `{out_root / 'model_comparison.csv'}`")
    lines.append("- 每模型训练曲线图与混淆矩阵图: `output/report/figures/`")
    lines.append("- 每模型原始详细指标: `output/<model>/metrics.json`")

    lines.append("")
    lines.append("## 4. 各模型图像文件")
    lines.append("")
    for r in rows:
        lines.append(f"### {r['model']}")
        lines.append(f"- 训练曲线: `{r['curve_png']}`")
        lines.append(f"- 混淆矩阵: `{r['cm_png']}`")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("Generated report files:")
    print(csv_path)
    print(md_path)
    print(fig_root)


if __name__ == "__main__":
    main()
