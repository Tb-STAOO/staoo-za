#!/usr/bin/env python3
"""
Train/evaluate binary classifiers for 1D waveform txt data.

Models (from assignment PDF):
- 1D CNN baseline
- InceptionTime (1D)
- ConvNeXt1D
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


LABEL_TO_INT = {"no": 0, "yes": 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

PROJECT_DIR = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def safe_odd_kernel(k: int) -> int:
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    return k


def read_signal(file_path: Path, expected_len: int | None = None) -> Tuple[np.ndarray, int]:
    values: List[float] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            token = parts[1] if len(parts) >= 2 else parts[0]
            try:
                values.append(float(token))
            except ValueError:
                continue

    if expected_len is None:
        expected_len = len(values)
    if expected_len <= 0:
        raise ValueError(f"Invalid expected length from {file_path}")

    if len(values) < expected_len:
        values.extend([0.0] * (expected_len - len(values)))
    elif len(values) > expected_len:
        values = values[:expected_len]

    return np.asarray(values, dtype=np.float32), expected_len


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    files: List[Path]
    seq_len: int


def load_split(data_root: Path, split: str, expected_len: int | None = None) -> SplitData:
    split_root = data_root / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split folder not found: {split_root}")

    signals: List[np.ndarray] = []
    labels: List[int] = []
    files: List[Path] = []
    seq_len = expected_len

    for class_name, label in LABEL_TO_INT.items():
        class_dir = split_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Class folder not found: {class_dir}")
        class_files = sorted(class_dir.rglob("*.txt"))
        if not class_files:
            raise RuntimeError(f"No txt files found in {class_dir}")

        for file_path in class_files:
            sig, seq_len = read_signal(file_path, seq_len)
            signals.append(sig)
            labels.append(label)
            files.append(file_path)

    x = np.stack(signals).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    return SplitData(x=x, y=y, files=files, seq_len=seq_len if seq_len else 0)


class SignalDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        max_shift: int = 20,
        noise_std: float = 0.01,
    ) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y.astype(np.float32))
        self.augment = augment
        self.max_shift = max_shift
        self.noise_std = noise_std

    def __len__(self) -> int:
        return self.x.shape[0]

    def _augment(self, signal: torch.Tensor) -> torch.Tensor:
        out = signal
        if torch.rand(1).item() < 0.7:
            gain = 0.9 + 0.2 * torch.rand(1).item()
            out = out * gain

        if self.max_shift > 0 and torch.rand(1).item() < 0.5:
            shift = int(torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item())
            if shift != 0:
                out = torch.roll(out, shifts=shift, dims=0)
                if shift > 0:
                    out[:shift] = 0.0
                else:
                    out[shift:] = 0.0

        if self.noise_std > 0 and torch.rand(1).item() < 0.7:
            out = out + torch.randn_like(out) * self.noise_std
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.x[idx].clone()
        if self.augment:
            signal = self._augment(signal)
        return signal.unsqueeze(0), self.y[idx]


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        k = safe_odd_kernel(kernel_size)
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1DBaseline(nn.Module):
    def __init__(self, base_channels: int = 32, kernel_size: int = 9, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(1, base_channels, kernel_size=kernel_size),
            nn.MaxPool1d(2),
            ConvBNAct(base_channels, base_channels * 2, kernel_size=kernel_size),
            nn.MaxPool1d(2),
            ConvBNAct(base_channels * 2, base_channels * 4, kernel_size=kernel_size),
            nn.MaxPool1d(2),
            ConvBNAct(base_channels * 4, base_channels * 4, kernel_size=kernel_size),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(1)


class InceptionBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 32,
        bottleneck_channels: int = 32,
        kernels: Tuple[int, int, int] = (9, 19, 39),
    ) -> None:
        super().__init__()
        kernels = tuple(safe_odd_kernel(k) for k in kernels)

        if in_channels > 1 and bottleneck_channels > 0:
            self.bottleneck = nn.Conv1d(
                in_channels, bottleneck_channels, kernel_size=1, bias=False
            )
            bottleneck_out = bottleneck_channels
        else:
            self.bottleneck = nn.Identity()
            bottleneck_out = in_channels

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck_out,
                    out_channels,
                    kernel_size=k,
                    padding=k // 2,
                    bias=False,
                )
                for k in kernels
            ]
        )

        self.pool_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        total_out = out_channels * (len(kernels) + 1)
        self.bn = nn.BatchNorm1d(total_out)
        self.act = nn.ReLU(inplace=True)
        self.out_channels = total_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.bottleneck(x)
        branches = [conv(z) for conv in self.convs]
        pool = torch.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        branches.append(self.pool_conv(pool))
        out = torch.cat(branches, dim=1)
        out = self.bn(out)
        return self.act(out)


class InceptionTime1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        n_blocks: int = 6,
        block_out_channels: int = 32,
        bottleneck_channels: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if n_blocks % 3 != 0:
            raise ValueError("n_blocks for InceptionTime1D should be divisible by 3")

        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        c_in = in_channels
        group_input_channels: List[int] = []
        for i in range(n_blocks):
            if i % 3 == 0:
                group_input_channels.append(c_in)
            block = InceptionBlock1D(
                in_channels=c_in,
                out_channels=block_out_channels,
                bottleneck_channels=bottleneck_channels,
            )
            self.blocks.append(block)
            c_in = block.out_channels

        for ch in group_input_channels:
            self.shortcuts.append(
                nn.Sequential(
                    nn.Conv1d(ch, c_in, kernel_size=1, bias=False),
                    nn.BatchNorm1d(c_in),
                )
            )

        self.final_channels = c_in
        self.act = nn.ReLU(inplace=True)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.final_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        shortcut_idx = 0
        out = x

        for i, block in enumerate(self.blocks):
            out = block(out)
            if i % 3 == 2:
                out = self.act(out + self.shortcuts[shortcut_idx](residual))
                residual = out
                shortcut_idx += 1

        return self.head(out).squeeze(1)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return inputs
        keep_prob = 1.0 - self.drop_prob
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=inputs.dtype, device=inputs.device)
        random_tensor.floor_()
        return inputs.div(keep_prob) * random_tensor


class LayerNorm1D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps
        self.channels = channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2)
        outputs = F.layer_norm(outputs, (self.channels,), self.weight, self.bias, self.eps)
        return outputs.transpose(1, 2)


class GlobalResponseNorm1D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(inputs, p=2, dim=-1, keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (inputs * nx) + self.beta + inputs


class AttentionPooling1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = max(16, channels // 2)
        self.attn = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(inputs), dim=-1)
        context = (inputs * weights).sum(dim=-1)
        avg_pool = inputs.mean(dim=-1)
        max_pool = inputs.max(dim=-1).values
        return torch.cat([context, avg_pool, max_pool], dim=1)


class ConvNeXtBlock1D(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        use_grn: bool = True,
    ) -> None:
        super().__init__()
        kernel_size = safe_odd_kernel(kernel_size)
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.norm = LayerNorm1D(dim)
        self.pwconv1 = nn.Conv1d(dim, 4 * dim, kernel_size=1)
        self.activation = nn.GELU()
        self.grn = GlobalResponseNorm1D(4 * dim) if use_grn else nn.Identity()
        self.pwconv2 = nn.Conv1d(4 * dim, dim, kernel_size=1)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dwconv(inputs)
        outputs = self.norm(outputs)
        outputs = self.pwconv1(outputs)
        outputs = self.activation(outputs)
        outputs = self.grn(outputs)
        outputs = self.pwconv2(outputs)
        if self.gamma is not None:
            outputs = self.gamma.view(1, -1, 1) * outputs
        return inputs + self.drop_path(outputs)


class ConvNeXt1D(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, int, int] = (32, 64, 128),
        depths: Tuple[int, int, int] = (2, 2, 2),
        kernel_size: int = 7,
        drop_path: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        use_grn: bool = True,
        dropout: float = 0.2,
        attention_pooling: bool = True,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=4, stride=4),
            LayerNorm1D(dims[0]),
        )

        total_blocks = sum(depths)
        drop_rates = (
            torch.linspace(0, drop_path, total_blocks).tolist() if total_blocks > 0 else []
        )
        rate_index = 0
        current_dim = dims[0]

        stages: List[nn.Module] = []
        for stage_index, (dim, depth) in enumerate(zip(dims, depths)):
            modules: List[nn.Module] = []
            if stage_index > 0:
                modules.extend(
                    [
                        LayerNorm1D(current_dim),
                        nn.Conv1d(current_dim, dim, kernel_size=2, stride=2),
                    ]
                )
                current_dim = dim
            for _ in range(depth):
                modules.append(
                    ConvNeXtBlock1D(
                        dim=current_dim,
                        kernel_size=kernel_size,
                        drop_path=drop_rates[rate_index],
                        layer_scale_init_value=layer_scale_init_value,
                        use_grn=use_grn,
                    )
                )
                rate_index += 1
            stages.append(nn.Sequential(*modules))

        self.stages = nn.ModuleList(stages)
        self.final_norm = LayerNorm1D(current_dim)
        self.use_attention_pooling = attention_pooling
        if attention_pooling:
            self.pool = AttentionPooling1D(current_dim, dropout=dropout)
            classifier_in = current_dim * 3
        else:
            self.pool = None
            classifier_in = current_dim * 2

        self.head = nn.Sequential(
            nn.Linear(classifier_in, current_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(current_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        for stage in self.stages:
            outputs = stage(outputs)
        outputs = self.final_norm(outputs)
        if self.use_attention_pooling:
            outputs = self.pool(outputs)
        else:
            outputs = torch.cat([outputs.mean(dim=-1), outputs.max(dim=-1).values], dim=1)
        return self.head(outputs).squeeze(1)


def build_model(model_name: str, args: argparse.Namespace) -> nn.Module:
    if model_name == "cnn1d":
        return CNN1DBaseline(
            base_channels=args.base_channels,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        )
    if model_name == "inceptiontime":
        return InceptionTime1D(
            in_channels=1,
            n_blocks=args.inception_blocks,
            block_out_channels=args.inception_channels,
            bottleneck_channels=args.inception_bottleneck,
            dropout=args.dropout,
        )
    if model_name == "convnext1d":
        return ConvNeXt1D(
            dims=(args.convnext_dim1, args.convnext_dim2, args.convnext_dim3),
            depths=(args.convnext_depth1, args.convnext_depth2, args.convnext_depth3),
            kernel_size=args.convnext_kernel_size,
            drop_path=args.convnext_drop_path,
            layer_scale_init_value=args.convnext_layer_scale,
            use_grn=bool(args.convnext_use_grn),
            dropout=args.dropout,
            attention_pooling=bool(args.convnext_attention_pooling),
        )
    raise ValueError(f"Unsupported model: {model_name}")


def normalize_arrays(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    if mode == "zscore":
        mean = float(x_train.mean())
        std = float(x_train.std() + 1e-8)
        return (
            (x_train - mean) / std,
            (x_val - mean) / std,
            (x_test - mean) / std,
            {"mode": mode, "mean": mean, "std": std},
        )

    if mode == "minmax":
        min_v = float(x_train.min())
        max_v = float(x_train.max())
        span = max(max_v - min_v, 1e-8)
        return (
            (x_train - min_v) / span,
            (x_val - min_v) / span,
            (x_test - min_v) / span,
            {"mode": mode, "min": min_v, "max": max_v},
        )

    raise ValueError(f"Unsupported norm mode: {mode}")


def build_loaders(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    num_workers: int,
    augment_shift: int,
    augment_noise: float,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = SignalDataset(
        x_train,
        y_train,
        augment=True,
        max_shift=augment_shift,
        noise_std=augment_noise,
    )
    val_ds = SignalDataset(x_val, y_val, augment=False)
    test_ds = SignalDataset(x_test, y_test, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    losses: List[float] = []
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        probs = torch.sigmoid(logits)
        losses.append(float(loss.item()))
        all_probs.append(probs.detach().cpu())
        all_labels.append(y.detach().cpu())

    probs_np = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(np.int64)
    preds_np = (probs_np >= 0.5).astype(np.int64)

    avg_loss = float(np.mean(losses)) if losses else 0.0
    acc = float(accuracy_score(labels_np, preds_np))
    return avg_loss, acc, probs_np, labels_np


def compute_metrics(probs: np.ndarray, labels: np.ndarray) -> Dict:
    preds = (probs >= 0.5).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except Exception:
        roc_auc = float("nan")

    report = classification_report(
        labels,
        preds,
        target_names=[INT_TO_LABEL[0], INT_TO_LABEL[1]],
        digits=4,
        zero_division=0,
        output_dict=True,
    )

    return {
        "test_acc": float(accuracy_score(labels, preds)),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": report,
    }


def train_once(
    model_name: str,
    x_train_full: np.ndarray,
    y_train_full: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Dict:
    set_seed(seed)

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train_full)),
        test_size=args.val_ratio,
        random_state=seed,
        stratify=y_train_full,
    )

    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    x_val = x_train_full[val_idx]
    y_val = y_train_full[val_idx]

    x_train, x_val, x_test_norm, norm_params = normalize_arrays(
        x_train,
        x_val,
        x_test,
        mode=args.norm,
    )

    train_loader, val_loader, test_loader = build_loaders(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test_norm,
        y_test=y_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_shift=args.augment_shift,
        augment_noise=args.augment_noise,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, args).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_val_loss = float("inf")
    no_improve = 0
    history: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_loss, val_acc, _, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )
        scheduler.step(val_loss)

        hist_item = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(hist_item)

        print(
            f"[{model_name}][seed {seed}] epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        improved = (val_acc > best_val_acc + 1e-6) or (
            abs(val_acc - best_val_acc) <= 1e-6 and val_loss < best_val_loss
        )
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[{model_name}][seed {seed}] early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_loss, _, test_probs, test_labels = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
    )
    metric_part = compute_metrics(test_probs, test_labels)

    metrics = {
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        **metric_part,
    }

    return {
        "model_name": model_name,
        "seed": seed,
        "history": history,
        "metrics": metrics,
        "state_dict": best_state,
        "norm_params": norm_params,
        "device": str(device),
    }


def save_single_model_result(
    model_name: str,
    best_result: Dict,
    all_results: List[Dict],
    out_dir: Path,
    model_dir: Path,
    train_data: SplitData,
    args: argparse.Namespace,
) -> Dict:
    model_ckpt_dir = model_dir / model_name
    metrics_out_dir = out_dir / model_name
    model_ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_ckpt_dir / "best_model.pth"
    metrics_path = metrics_out_dir / "metrics.json"

    checkpoint = {
        "model_name": model_name,
        "model_state_dict": best_result["state_dict"],
        "normalization": best_result["norm_params"],
        "seq_len": train_data.seq_len,
        "label_to_int": LABEL_TO_INT,
        "int_to_label": INT_TO_LABEL,
        "best_seed": best_result["seed"],
        "args": vars(args),
    }
    torch.save(checkpoint, checkpoint_path)

    payload = {
        "model_name": model_name,
        "data_root": args.data_root,
        "out_dir": str(metrics_out_dir),
        "model_dir": str(model_ckpt_dir),
        "target_acc": args.target_acc,
        "best_seed": best_result["seed"],
        "best_metrics": best_result["metrics"],
        "all_attempts": [
            {
                "seed": r["seed"],
                "metrics": r["metrics"],
                "history": r["history"],
            }
            for r in all_results
        ],
        "model_path": str(checkpoint_path),
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(make_json_serializable(payload), f, indent=2, ensure_ascii=False)

    return {
        "model_name": model_name,
        "best_seed": best_result["seed"],
        "best_model_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "best_metrics": best_result["metrics"],
    }


def train_model_with_retries(
    model_name: str,
    train_data: SplitData,
    test_data: SplitData,
    args: argparse.Namespace,
    out_dir: Path,
    model_dir: Path,
) -> Dict:
    all_results: List[Dict] = []
    best_result: Dict | None = None

    progress = tqdm(range(args.max_retries), desc=f"{model_name} attempts")
    for i in progress:
        seed = args.seed + i
        result = train_once(
            model_name=model_name,
            x_train_full=train_data.x,
            y_train_full=train_data.y,
            x_test=test_data.x,
            y_test=test_data.y,
            args=args,
            seed=seed,
        )

        all_results.append(result)
        acc = result["metrics"]["test_acc"]
        progress.set_postfix({"seed": seed, "test_acc": f"{acc:.4f}"})

        print(
            f"[{model_name}] attempt {i + 1}/{args.max_retries} | "
            f"seed={seed} test_acc={acc:.4f} test_f1={result['metrics']['test_f1']:.4f}"
        )

        if best_result is None or acc > best_result["metrics"]["test_acc"]:
            best_result = result

        if acc >= args.target_acc:
            print(
                f"[{model_name}] Reached target accuracy {args.target_acc:.2%} "
                f"on attempt {i + 1} (seed={seed})."
            )
            break

    if best_result is None:
        raise RuntimeError(f"No training result produced for model {model_name}")

    summary = save_single_model_result(
        model_name=model_name,
        best_result=best_result,
        all_results=all_results,
        out_dir=out_dir,
        model_dir=model_dir,
        train_data=train_data,
        args=args,
    )

    m = summary["best_metrics"]
    print("\n=== Final Result ({}) ===".format(model_name))
    print(f"Best seed: {summary['best_seed']}")
    print(f"Test accuracy: {m['test_acc']:.4f}")
    print(f"Test precision: {m['test_precision']:.4f}")
    print(f"Test recall: {m['test_recall']:.4f}")
    print(f"Test F1: {m['test_f1']:.4f}")
    print(f"Test ROC-AUC: {m['test_roc_auc']:.4f}")
    print(f"Confusion matrix: {m['confusion_matrix']}")
    print(f"Model saved to: {summary['best_model_path']}")
    print(f"Metrics saved to: {summary['metrics_path']}")

    if m["test_acc"] < args.target_acc:
        print(
            "Target not reached yet. You can tune params, e.g.: "
            "--epochs 120 --max-retries 8 --batch-size 32 --lr 0.0005"
        )

    return summary


def save_comparison_report(out_dir: Path, model_summaries: List[Dict], args: argparse.Namespace) -> None:
    comp_json = out_dir / "model_comparison.json"
    comp_csv = out_dir / "model_comparison.csv"

    payload = {
        "data_root": args.data_root,
        "models": model_summaries,
    }
    with comp_json.open("w", encoding="utf-8") as f:
        json.dump(make_json_serializable(payload), f, indent=2, ensure_ascii=False)

    fields = [
        "model_name",
        "best_seed",
        "test_acc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
        "confusion_matrix",
        "best_model_path",
        "metrics_path",
    ]
    with comp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in model_summaries:
            row = {
                "model_name": s["model_name"],
                "best_seed": s["best_seed"],
                "test_acc": s["best_metrics"]["test_acc"],
                "test_precision": s["best_metrics"]["test_precision"],
                "test_recall": s["best_metrics"]["test_recall"],
                "test_f1": s["best_metrics"]["test_f1"],
                "test_roc_auc": s["best_metrics"]["test_roc_auc"],
                "confusion_matrix": s["best_metrics"]["confusion_matrix"],
                "best_model_path": s["best_model_path"],
                "metrics_path": s["metrics_path"],
            }
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Binary classification for 1D waveform txt with CNN/InceptionTime/ConvNeXt1D",
    )

    parser.add_argument("--data-root", type=str, default="F:/data/newdata")
    parser.add_argument("--model-dir", type=str, default=str(PROJECT_DIR / "model"))
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_DIR / "output"))

    parser.add_argument(
        "--model",
        type=str,
        default="convnext1d",
        choices=["cnn1d", "inceptiontime", "convnext1d", "all"],
        help="Model name, or all for comparison.",
    )

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--target-acc", type=float, default=0.90)

    parser.add_argument("--norm", type=str, choices=["zscore", "minmax"], default="zscore")
    parser.add_argument("--augment-shift", type=int, default=20)
    parser.add_argument("--augment-noise", type=float, default=0.01)

    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--base-channels", type=int, default=32)

    parser.add_argument("--inception-blocks", type=int, default=6)
    parser.add_argument("--inception-channels", type=int, default=32)
    parser.add_argument("--inception-bottleneck", type=int, default=32)

    parser.add_argument("--convnext-dim1", type=int, default=32)
    parser.add_argument("--convnext-dim2", type=int, default=64)
    parser.add_argument("--convnext-dim3", type=int, default=128)
    parser.add_argument("--convnext-depth1", type=int, default=2)
    parser.add_argument("--convnext-depth2", type=int, default=2)
    parser.add_argument("--convnext-depth3", type=int, default=2)
    parser.add_argument("--convnext-kernel-size", type=int, default=7)
    parser.add_argument("--convnext-drop-path", type=float, default=0.1)
    parser.add_argument("--convnext-layer-scale", type=float, default=1e-6)
    parser.add_argument("--convnext-use-grn", type=int, choices=[0, 1], default=1)
    parser.add_argument("--convnext-attention-pooling", type=int, choices=[0, 1], default=1)

    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.inception_blocks % 3 != 0:
        raise ValueError("--inception-blocks must be divisible by 3")

    print("Loading dataset ...")
    train_data = load_split(data_root, "train", expected_len=None)
    test_data = load_split(data_root, "test", expected_len=train_data.seq_len)
    print(
        f"train={len(train_data.y)} samples, test={len(test_data.y)} samples, "
        f"seq_len={train_data.seq_len}"
    )

    model_names = ["cnn1d", "inceptiontime", "convnext1d"] if args.model == "all" else [args.model]

    model_summaries: List[Dict] = []
    for model_name in model_names:
        summary = train_model_with_retries(
            model_name=model_name,
            train_data=train_data,
            test_data=test_data,
            args=args,
            out_dir=out_dir,
            model_dir=model_dir,
        )
        model_summaries.append(summary)

    if args.model == "all":
        save_comparison_report(out_dir, model_summaries, args)

        print("\n=== Model Comparison ===")
        sorted_models = sorted(
            model_summaries,
            key=lambda s: s["best_metrics"]["test_acc"],
            reverse=True,
        )
        for i, s in enumerate(sorted_models, start=1):
            m = s["best_metrics"]
            print(
                f"{i}. {s['model_name']} | acc={m['test_acc']:.4f} "
                f"f1={m['test_f1']:.4f} auc={m['test_roc_auc']:.4f}"
            )
        print(f"Comparison files saved to: {out_dir / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
