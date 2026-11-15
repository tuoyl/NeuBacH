#!/usr/bin/env python3
"""
Train a transformer-based model to predict the HXMT HE background spectrum.

The dataset is grouped into non-overlapping 32-second windows (configurable)
to suppress per-second statistical noise. Each training sample contains the
sequence of per-second features inside the window, while the regression target
is the summed NORMAL_COUNTS spectrum over the same interval. Channel ranges
used for regression can be constrained via --channel-min/--channel-max, and
specific channel bands (e.g., emission lines) can be up-weighted during training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


# -----------------------------
# Utility helpers
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_csv(path: Path, limit_rows: Optional[int] = None) -> pd.DataFrame:
    kwargs = {"nrows": limit_rows} if limit_rows else {}
    df = pd.read_csv(path, **kwargs)
    return df.copy()


def find_target_columns(
    df: pd.DataFrame,
    prefix: str = "NORMAL_COUNTS_CHANNEL_",
    ch_min: Optional[int] = None,
    ch_max: Optional[int] = None,
) -> List[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    cols.sort(key=lambda x: int(x.split("_")[-1]))
    if not cols:
        raise ValueError(f"No target columns with prefix '{prefix}' were found.")

    ch_min = 0 if ch_min is None else ch_min
    ch_max = 255 if ch_max is None else ch_max
    if ch_min < 0 or ch_max > 255 or ch_min > ch_max:
        raise ValueError(f"Invalid channel range [{ch_min}, {ch_max}].")

    filtered = []
    for col in cols:
        try:
            idx = int(col.split("_")[-1])
        except ValueError:
            continue
        if ch_min <= idx <= ch_max:
            filtered.append(col)

    if not filtered:
        raise ValueError(
            f"No target columns fall within the requested channel range [{ch_min}, {ch_max}]."
        )
    return filtered


def find_feature_columns(df: pd.DataFrame, target_cols: Sequence[str]) -> List[str]:
    feats = [c for c in df.columns if c not in target_cols]
    feats = [c for c in feats if df[c].dtype.kind in "if"]
    if not feats:
        raise ValueError("No numeric feature columns found.")
    return feats


def parse_channel_ranges(ranges: Optional[str]) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    out: List[Tuple[int, int]] = []
    for part in ranges.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' not in part:
            raise ValueError(f"Invalid channel range '{part}'. Expected format like '8-14'.")
        a, b = part.split('-', 1)
        try:
            lo = int(a)
            hi = int(b)
        except ValueError as exc:
            raise ValueError(f"Invalid integers in channel range '{part}'.") from exc
        if lo > hi:
            lo, hi = hi, lo
        if lo < 0 or hi > 255:
            raise ValueError(f"Channel range '{part}' is outside [0,255].")
        out.append((lo, hi))
    return out


def assign_segments(met: np.ndarray, max_gap: float) -> np.ndarray:
    if len(met) == 0:
        return np.array([], dtype=np.int64)
    segments = np.zeros_like(met, dtype=np.int64)
    gaps = np.diff(met, prepend=met[0])
    seg_id = 0
    for i in range(1, len(met)):
        if gaps[i] > max_gap:
            seg_id += 1
        segments[i] = seg_id
    return segments


def compute_window_starts(
    segments: np.ndarray,
    window: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    starts: List[int] = []
    start_segments: List[int] = []
    if len(segments) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    unique_segments = np.unique(segments)
    for seg in unique_segments:
        idx = np.nonzero(segments == seg)[0]
        if len(idx) < window:
            continue
        local_count = len(idx) - window + 1
        for offset in range(0, local_count, stride):
            start = idx[offset]
            starts.append(start)
            start_segments.append(int(seg))
    return np.asarray(starts, dtype=np.int64), np.asarray(start_segments, dtype=np.int64)


def channel_from_name(name: str) -> int:
    try:
        return int(name.split('_')[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unable to parse channel index from column '{name}'.") from exc


def feature_stats(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0)
    sigma = x.std(axis=0)
    sigma = np.clip(sigma, 1e-6, None)
    return mu.astype(np.float32), sigma.astype(np.float32)


def split_segment_indices(
    sample_segments: np.ndarray,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    segments = np.unique(sample_segments)
    total_segments = len(segments)
    if total_segments == 0:
        return [], [], []

    seg_list = list(segments)
    rng = random.Random(seed)
    rng.shuffle(seg_list)

    def desired(count: float) -> int:
        return int(round(total_segments * count))

    val_count = desired(val_ratio)
    test_count = desired(test_ratio)

    if total_segments >= 3:
        val_count = max(1, val_count)
        test_count = max(1, test_count)
    elif total_segments == 2:
        val_count = 1
        test_count = 0
    else:
        val_count = 0
        test_count = 0

    while val_count + test_count >= total_segments and total_segments > 1:
        if val_count >= test_count and val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            break

    train_count = total_segments - val_count - test_count
    if train_count <= 0:
        if val_count > 1:
            val_count -= 1
        elif test_count > 1:
            test_count -= 1
        train_count = total_segments - val_count - test_count
        if train_count <= 0:
            train_count = total_segments
            val_count = 0
            test_count = 0

    val_segments = set(seg_list[:val_count])
    test_segments = set(seg_list[val_count:val_count + test_count])
    train_segments = set(seg_list[val_count + test_count:])

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for idx, seg in enumerate(sample_segments):
        if seg in train_segments:
            train_idx.append(idx)
        elif seg in val_segments:
            val_idx.append(idx)
        else:
            test_idx.append(idx)

    if not train_idx and val_idx:
        train_idx, val_idx = val_idx, train_idx
    if not train_idx and test_idx:
        train_idx, test_idx = test_idx, train_idx

    return train_idx, val_idx, test_idx


# -----------------------------
# Dataset
# -----------------------------

class HXMTWindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        target_cols: Sequence[str],
        window: int = 32,
        stride: int = 32,
        max_gap: float = 10.0,
        mu: Optional[np.ndarray] = None,
        sigma: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        if "MET" not in df.columns:
            raise ValueError("Input dataframe must contain a 'MET' column for time ordering.")

        self.window = window
        self.stride = stride
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols)
        self.max_gap = max_gap

        df_sorted = df.sort_values("MET").reset_index(drop=True)
        self.features = df_sorted[self.feature_cols].to_numpy(dtype=np.float32)
        self.targets = df_sorted[self.target_cols].to_numpy(dtype=np.float32)
        self.met = df_sorted["MET"].to_numpy(dtype=np.float64)

        self.segment_ids = assign_segments(self.met, max_gap=max_gap)
        self.window_starts, self.sample_segments = compute_window_starts(
            self.segment_ids, window=self.window, stride=self.stride
        )

        if len(self.window_starts) == 0:
            raise ValueError(
                "No valid windows were found. Consider lowering the stride or max_gap, "
                "or inspect the input data."
            )

        if mu is None or sigma is None:
            mu, sigma = feature_stats(self.features)
        self.mu = mu.astype(np.float32)
        self.sigma = sigma.astype(np.float32)

        # Pre-compute metadata per sample for evaluation/export.
        self.sample_met_start = self.met[self.window_starts]
        self.sample_met_end = self.met[self.window_starts + self.window - 1]

    def __len__(self) -> int:
        return len(self.window_starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.window_starts[idx]
        end = start + self.window
        x = self.features[start:end]
        y = self.targets[start:end].sum(axis=0, dtype=np.float32)
        x = (x - self.mu) / self.sigma
        return torch.from_numpy(x), torch.from_numpy(y)

    def metadata(self, idx: int) -> Dict[str, float]:
        return {
            "segment": int(self.sample_segments[idx]),
            "met_start": float(self.sample_met_start[idx]),
            "met_end": float(self.sample_met_end[idx]),
        }


# -----------------------------
# Model
# -----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            div_term = div_term[:-1]
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.pe[:, :length]


class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional = PositionalEncoding(d_model=d_model, max_len=2048)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, target_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.positional(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)


# -----------------------------
# Training / evaluation
# -----------------------------

@dataclass
class EpochMetrics:
    loss: float
    mae: float
    rmse: float


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[float, float]:
    diff = preds - targets
    mae = diff.abs().mean().item()
    rmse = torch.sqrt((diff ** 2).mean()).item()
    return mae, rmse


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    use_log_target: bool,
    grad_clip: Optional[float],
    channel_weights: Optional[torch.Tensor],
    train: bool = True,
) -> EpochMetrics:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_samples = 0

    context = torch.enable_grad if train else torch.no_grad

    with context():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            if use_log_target:
                target = torch.log1p(y)
                preds = torch.expm1(logits)
            else:
                target = y
                preds = logits

            loss_raw = criterion(logits, target)
            if channel_weights is not None:
                loss = (loss_raw * channel_weights).mean()
            else:
                loss = loss_raw.mean()

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            mae, rmse = compute_metrics(preds, y)
            batch_size = x.size(0)
            total_loss += loss.item() * batch_size
            total_mae += mae * batch_size
            total_rmse += rmse * batch_size
            total_samples += batch_size

    total_samples = max(total_samples, 1)
    return EpochMetrics(
        loss=total_loss / total_samples,
        mae=total_mae / total_samples,
        rmse=total_rmse / total_samples,
    )


def evaluate_dataset(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    use_log_target: bool,
    batch_size: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if use_log_target:
                preds = torch.expm1(logits)
            else:
                preds = logits
            preds_list.append(preds.cpu().numpy())
            targets_list.append(y.numpy())

    preds_all = np.concatenate(preds_list, axis=0)
    targets_all = np.concatenate(targets_list, axis=0) if targets_list else None
    return preds_all, targets_all


# -----------------------------
# Main execution
# -----------------------------

def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=data_dir / "hxmt_merged_dataset_HE_v3.0.csv")
    parser.add_argument("--window", type=int, default=32, help="Window length in seconds.")
    parser.add_argument("--stride", type=int, default=32, help="Stride between windows.")
    parser.add_argument("--max-gap", type=float, default=10.0, help="Maximum MET gap (s) inside a segment.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional row cap for quick experiments.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio by segment.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio by segment.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default=None, help="Override training device (cpu|cuda|mps).")
    parser.add_argument("--target-log1p", action="store_true", help="Train on log1p(target) instead of counts.")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (epochs).")
    parser.add_argument("--min-delta", type=float, default=1e-4, help="Required improvement for early stopping.")
    parser.add_argument("--save-dir", type=Path, default=script_dir / "artifacts")
    parser.add_argument("--export-predictions", type=Path, default=None, help="Optional CSV to save test predictions.")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Optional path to dump metrics JSON.")
    parser.add_argument("--channel-min", type=int, default=0, help="Minimum detector channel to include (inclusive).")
    parser.add_argument("--channel-max", type=int, default=255, help="Maximum detector channel to include (inclusive).")
    parser.add_argument(
        "--focus-channel-ranges",
        type=str,
        default="8-14,30-40,110-140",
        help="Comma-separated channel ranges to up-weight in the loss (e.g. '8-14,30-40'). Empty to disable.",
    )
    parser.add_argument(
        "--focus-weight",
        type=float,
        default=3.0,
        help="Multiplicative weight applied to channels inside the focus ranges.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else auto_device()
    print(f"Using device: {device}")

    train_path = args.train_csv.expanduser()

    if not train_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")

    print(f"Loading train data from {train_path}")
    train_df = read_csv(train_path, limit_rows=args.limit_rows)
    target_cols = find_target_columns(
        train_df,
        ch_min=args.channel_min,
        ch_max=args.channel_max,
    )
    feature_cols = find_feature_columns(train_df, target_cols)

    mu, sigma = feature_stats(train_df[feature_cols].to_numpy(dtype=np.float32))

    train_dataset_full = HXMTWindowDataset(
        train_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window=args.window,
        stride=args.stride,
        max_gap=args.max_gap,
        mu=mu,
        sigma=sigma,
    )

    train_idx, val_idx, test_idx = split_segment_indices(
        train_dataset_full.sample_segments,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    train_subset = Subset(train_dataset_full, train_idx)
    val_subset = Subset(train_dataset_full, val_idx) if len(val_idx) > 0 else None
    test_subset = Subset(train_dataset_full, test_idx) if len(test_idx) > 0 else None

    msg = f"Train samples: {len(train_subset)}"
    if val_subset:
        msg += f", Val samples: {len(val_subset)}"
    if test_subset:
        msg += f", Test samples: {len(test_subset)}"
    print(msg)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = (
        DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        if val_subset
        else None
    )
    test_loader = (
        DataLoader(
            test_subset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        if test_subset
        else None
    )

    model = TransformerRegressor(
        input_dim=len(feature_cols),
        target_dim=len(target_cols),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
    ).to(device)

    focus_ranges = parse_channel_ranges(args.focus_channel_ranges)
    focus_weight = float(args.focus_weight)
    channel_indices = np.array([channel_from_name(c) for c in target_cols], dtype=np.int32)
    weight_vec = np.ones(len(target_cols), dtype=np.float32)
    if focus_ranges and focus_weight != 1.0:
        focus_mask = np.zeros_like(weight_vec, dtype=bool)
        for lo, hi in focus_ranges:
            focus_mask |= (channel_indices >= lo) & (channel_indices <= hi)
        if focus_mask.any():
            weight_vec[focus_mask] = focus_weight
            weight_vec /= weight_vec.mean()
            print(
                "Applying channel focus:",
                f"ranges={focus_ranges} weight={focus_weight} (normalized mean=1.0)",
            )
        else:
            print("Warning: focus ranges provided but no target channels matched; skipping focus weighting.")
            focus_ranges = []
    weights_are_uniform = np.allclose(weight_vec, 1.0)
    channel_weights_tensor = None if weights_are_uniform else torch.from_numpy(weight_vec).to(device).unsqueeze(0)
    criterion = nn.MSELoss(reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_path = Path(args.save_dir) / "best_model.pt"

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            use_log_target=args.target_log1p,
            grad_clip=args.grad_clip,
            channel_weights=channel_weights_tensor,
            train=True,
        )
        if val_loader is not None:
            val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                optimizer=None,
                device=device,
                use_log_target=args.target_log1p,
                grad_clip=None,
                channel_weights=channel_weights_tensor,
                train=False,
            )
        else:
            val_metrics = train_metrics

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_mae": train_metrics.mae,
                "train_rmse": train_metrics.rmse,
                "val_loss": val_metrics.loss,
                "val_mae": val_metrics.mae,
                "val_rmse": val_metrics.rmse,
            }
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics.loss:.4f} val_loss={val_metrics.loss:.4f} | "
            f"train_mae={train_metrics.mae:.4f} val_mae={val_metrics.mae:.4f}"
        )

        if val_metrics.loss + args.min_delta < best_val_loss:
            best_val_loss = val_metrics.loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "feature_cols": feature_cols,
                    "target_cols": target_cols,
                    "mu": mu,
                    "sigma": sigma,
                    "channel_weights": weight_vec.tolist(),
                    "args": vars(args),
                    "val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            print(f"  Saved checkpoint to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"Best validation loss {best_val_loss:.4f} at epoch {best_epoch}")

    if checkpoint_path.exists():
        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        print(f"Loaded best model from {checkpoint_path}")

    test_metrics: Optional[EpochMetrics] = None
    test_predictions: Optional[np.ndarray] = None
    test_targets: Optional[np.ndarray] = None

    if test_loader is not None:
        test_metrics = run_epoch(
            model,
            test_loader,
            criterion,
            optimizer=None,
            device=device,
            use_log_target=args.target_log1p,
            grad_clip=None,
            channel_weights=channel_weights_tensor,
            train=False,
        )
        print(
            f"Test loss={test_metrics.loss:.4f} "
            f"MAE={test_metrics.mae:.4f} RMSE={test_metrics.rmse:.4f}"
        )

        test_predictions, test_targets = evaluate_dataset(
            model,
            test_subset,
            device=device,
            use_log_target=args.target_log1p,
            batch_size=args.batch_size,
        )

        if args.export_predictions:
            if isinstance(test_subset, Subset):
                base_dataset = test_subset.dataset
                indices = np.asarray(test_subset.indices)
            else:
                base_dataset = test_subset
                indices = np.arange(len(test_subset))
            export_path = args.export_predictions.expanduser()
            export_df = pd.DataFrame(
                test_predictions,
                columns=base_dataset.target_cols,
            )
            export_df.insert(0, "MET_END", base_dataset.sample_met_end[indices])
            export_df.insert(0, "MET_START", base_dataset.sample_met_start[indices])
            if test_targets is not None:
                truth_df = pd.DataFrame(
                    test_targets,
                    columns=[f"{c}_true" for c in base_dataset.target_cols],
                )
                export_df = pd.concat([export_df, truth_df], axis=1)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            export_df.to_csv(export_path, index=False)
            print(f"Saved predictions to {export_path}")
    else:
        print("Test split is empty; skipping test evaluation.")

    if args.metrics_json:
        metrics_payload = {
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "history": history,
        }
        if test_metrics:
            metrics_payload["test"] = {
                "loss": test_metrics.loss,
                "mae": test_metrics.mae,
                "rmse": test_metrics.rmse,
            }
        metrics_path = args.metrics_json.expanduser()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)
        print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
