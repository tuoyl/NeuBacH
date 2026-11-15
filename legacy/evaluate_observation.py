#!/usr/bin/env python3
"""
Evaluate a trained HXMT background model on a single observation.

Given a CSV file with the same structure as the training dataset, the script:
  * Loads the transformer checkpoint produced by `train_transformer_hxmt_32s.py`
  * Generates model predictions over the entire observation
  * Plots the aggregated spectrum (observed vs predicted) with residuals
  * Plots the 1 s background lightcurve (observed vs predicted) with residuals
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_transformer_hxmt_32s import (
    HXMTWindowDataset,
    TransformerRegressor,
    auto_device,
    channel_from_name,
    read_csv,
    set_seed,
)


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)

    model_args = state.get("args", {})
    feature_cols = state["feature_cols"]
    target_cols = state["target_cols"]
    mu = np.asarray(state["mu"], dtype=np.float32)
    sigma = np.asarray(state["sigma"], dtype=np.float32)

    model = TransformerRegressor(
        input_dim=len(feature_cols),
        target_dim=len(target_cols),
        d_model=model_args.get("d_model", 128),
        nhead=model_args.get("nhead", 4),
        num_layers=model_args.get("layers", 4),
        dim_feedforward=model_args.get("ffn_dim", 256),
        dropout=model_args.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    target_log1p = bool(model_args.get("target_log1p", False))

    channel_weights = np.asarray(state.get("channel_weights"), dtype=np.float32) if "channel_weights" in state else None

    return model, feature_cols, target_cols, mu, sigma, target_log1p, model_args, channel_weights


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing[:10]} ...")


def predict_dataset(
    model: TransformerRegressor,
    dataset: HXMTWindowDataset,
    device: torch.device,
    target_log1p: bool,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            if target_log1p:
                preds_batch = torch.expm1(logits)
            else:
                preds_batch = logits
            preds.append(preds_batch.cpu().numpy())
            trues.append(y.numpy())

    if not preds:
        raise RuntimeError("Prediction dataset is empty; check observation length vs. model window.")

    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def aggregate_metrics(pred_windows: np.ndarray, obs_windows: np.ndarray) -> Dict[str, float]:
    pred_spectrum = pred_windows.sum(axis=0)
    obs_spectrum = obs_windows.sum(axis=0)
    residual = obs_spectrum - pred_spectrum

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    total_obs = float(obs_spectrum.sum())
    total_pred = float(pred_spectrum.sum())
    rel_error = float((total_pred - total_obs) / max(total_obs, 1e-6))

    return {
        "pred_spectrum": pred_spectrum,
        "obs_spectrum": obs_spectrum,
        "residual": residual,
        "mae": mae,
        "rmse": rmse,
        "total_obs": total_obs,
        "total_pred": total_pred,
        "rel_error": rel_error,
    }


def plot_spectrum(
    channels: np.ndarray,
    obs: np.ndarray,
    pred: np.ndarray,
    residual: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_spec, ax_res = axes

    ax_spec.errorbar(channels, obs, np.sqrt(obs), label="Observed", color="#1f77b4", linewidth=1.2, fmt='.')
    ax_spec.plot(channels, pred, label="Predicted", color="#ff7f0e", linewidth=1.2)
    ax_spec.set_ylabel("Counts (aggregated)")
    ax_spec.set_yscale("log")
    ax_spec.legend()
    ax_spec.grid(alpha=0.25)
    ax_spec.set_title(title)

    ax_res.errorbar(channels, residual, np.sqrt(obs), color="#2ca02c", linewidth=1.0, fmt='.')
    ax_res.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax_res.set_xlabel("Channel")
    ax_res.set_ylabel("Residual")
    ax_res.grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_lightcurve(
    met: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    valid_mask = ~np.isnan(predicted)
    residual = np.full_like(predicted, np.nan)
    residual[valid_mask] = observed[valid_mask] - predicted[valid_mask]

    time_rel = met - met[0]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_curve, ax_res = axes

    ax_curve.plot(time_rel, observed, label="Observed (1 s)", color="#1f77b4", linewidth=0.9)
    ax_curve.plot(time_rel, predicted, label="Predicted (1 s equivalent)", color="#ff7f0e", linewidth=1.1)
    ax_curve.set_ylabel("Counts / s")
    ax_curve.legend()
    ax_curve.grid(alpha=0.25)
    ax_curve.set_title(title)

    ax_res.plot(time_rel, residual, color="#2ca02c", linewidth=0.9)
    ax_res.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax_res.set_xlabel("Time since start (s)")
    ax_res.set_ylabel("Residual")
    ax_res.grid(alpha=0.25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir.parent / "data"
    default_checkpoint = script_dir / "artifacts" / "best_model.pt"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-csv", type=Path, required=True, help="Observation CSV to evaluate.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        help="Checkpoint produced by train_transformer_hxmt_32s.py.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Override model window length. Defaults to the value stored in the checkpoint.",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=None,
        help="Maximum MET gap defining a continuous segment. Defaults to checkpoint setting.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference.")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional row cap for quick tests.")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "artifacts", help="Output directory.")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for output filenames.")
    parser.add_argument("--metrics-json", type=Path, default=None, help="Optional JSON to store metrics.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda|cpu|mps).")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else auto_device()
    model, feature_cols, target_cols, mu, sigma, target_log1p, model_args, channel_weights = load_checkpoint(
        args.checkpoint, device
    )
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Using device: {device}")

    default_window = int(model_args.get("window", 32))
    window = args.window if args.window is not None else default_window
    if window != default_window:
        print(f"Warning: overriding checkpoint window ({default_window}) with {window}.")

    default_max_gap = float(model_args.get("max_gap", 10.0))
    max_gap = float(args.max_gap) if args.max_gap is not None else default_max_gap

    df = read_csv(args.data_csv.expanduser(), limit_rows=args.limit_rows)
    ensure_columns(df, list(feature_cols) + list(target_cols))

    if len(df) < window:
        raise RuntimeError(
            f"Observation has only {len(df)} rows, but the model expects sequences of length {window}."
        )

    dataset_stride = HXMTWindowDataset(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window=window,
        stride=window,
        max_gap=max_gap,
        mu=mu,
        sigma=sigma,
    )

    pred_windows, obs_windows = predict_dataset(
        model,
        dataset_stride,
        device=device,
        target_log1p=target_log1p,
        batch_size=args.batch_size,
    )
    metrics = aggregate_metrics(pred_windows, obs_windows)

    channels = np.array([channel_from_name(c) for c in target_cols], dtype=np.int32)
    observation_name = args.prefix or args.data_csv.stem
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    spectrum_path = output_dir / f"{observation_name}_spectrum.png"
    spectrum_title = (
        f"{observation_name}: Spectrum | "
        f"total_obs={metrics['total_obs']:.1f} total_pred={metrics['total_pred']:.1f} "
        f"MAE={metrics['mae']:.2f} RMSE={metrics['rmse']:.2f}"
    )
    plot_spectrum(
        channels,
        metrics["obs_spectrum"],
        metrics["pred_spectrum"],
        metrics["residual"],
        spectrum_path,
        spectrum_title,
    )
    print(f"Saved spectrum figure to {spectrum_path}")

    # Build per-second lightcurve comparison
    dataset_stride1 = HXMTWindowDataset(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window=window,
        stride=1,
        max_gap=max_gap,
        mu=mu,
        sigma=sigma,
    )

    pred_stride1, obs_stride1 = predict_dataset(
        model,
        dataset_stride1,
        device=device,
        target_log1p=target_log1p,
        batch_size=args.batch_size,
    )

    pred_total = pred_stride1.sum(axis=1)
    pred_per_second = pred_total / window

    observed_total_per_second = df[target_cols].sum(axis=1).to_numpy(dtype=np.float64)
    predicted_per_second_series = np.full_like(observed_total_per_second, np.nan, dtype=np.float64)

    indices_end = dataset_stride1.window_starts + window - 1
    predicted_per_second_series[indices_end] = pred_per_second

    met = df["MET"].to_numpy(dtype=np.float64)
    lightcurve_path = output_dir / f"{observation_name}_lightcurve.png"
    plot_lightcurve(
        met,
        observed_total_per_second,
        predicted_per_second_series,
        lightcurve_path,
        f"{observation_name}: Background lightcurve (window={window}s)",
    )
    print(f"Saved lightcurve figure to {lightcurve_path}")

    print(
        f"Observation metrics -> total_obs={metrics['total_obs']:.1f}, total_pred={metrics['total_pred']:.1f}, "
        f"rel_error={metrics['rel_error']*100:.2f}%, MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}"
    )

    if args.metrics_json:
        payload = {
            "observation": observation_name,
            "total_obs": metrics["total_obs"],
            "total_pred": metrics["total_pred"],
            "relative_error": metrics["rel_error"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "window": window,
            "max_gap": max_gap,
            "checkpoint": str(args.checkpoint),
            "channel_weights": channel_weights.tolist() if channel_weights is not None else None,
        }
        metrics_path = args.metrics_json.expanduser()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()

