#!/usr/bin/env python3
"""
Monthly evaluation plots for the HXMT transformer background model.

Given a trained checkpoint from `train_transformer_hxmt_32s.py`, the script:
  1. Samples multiple 1024-second (configurable) observation chunks spaced
     roughly one month apart across the full dataset.
  2. For each chunk, aggregates the observed background spectrum and the model
     prediction, saving comparison figures and recording summary metrics.

This helps verify that the model performance is stable over long timescales.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_transformer_hxmt_32s import (
    HXMTWindowDataset,
    TransformerRegressor,
    auto_device,
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

    return model, feature_cols, target_cols, mu, sigma, target_log1p


def ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing[:10]}")


def model_window_pass(
    model: TransformerRegressor,
    dataset: HXMTWindowDataset,
    indices: Sequence[int],
    device: torch.device,
    target_log1p: bool,
) -> np.ndarray:
    spectra = []
    with torch.no_grad():
        for idx in indices:
            x, _ = dataset[idx]
            pred = model(x.unsqueeze(0).to(device))
            if target_log1p:
                pred = torch.expm1(pred)
            spectra.append(pred.squeeze(0).cpu().numpy())
    return np.stack(spectra)


def observed_window_pass(dataset: HXMTWindowDataset, indices: Sequence[int]) -> np.ndarray:
    spectra = []
    for idx in indices:
        _, y = dataset[idx]
        spectra.append(y.numpy())
    return np.stack(spectra)


def is_contiguous_block(
    dataset: HXMTWindowDataset,
    block: Sequence[int],
) -> bool:
    segments = dataset.sample_segments[block]
    if not np.all(segments == segments[0]):
        return False
    starts = dataset.window_starts[block]
    return np.all(np.diff(starts) == dataset.window)


def find_monthly_blocks(
    dataset: HXMTWindowDataset,
    block_windows: int,
    month_stride_seconds: float,
    max_evals: int,
) -> List[np.ndarray]:
    if block_windows <= 0:
        raise ValueError("block_windows must be positive.")

    times = dataset.sample_met_start

    if len(times) < block_windows:
        return []

    blocks: List[np.ndarray] = []
    window = dataset.window
    time_limit = times[-1] - window * (block_windows - 1)
    target_time = times[0]

    while target_time <= time_limit and len(blocks) < max_evals:
        start_idx = int(np.searchsorted(times, target_time, side="left"))
        found = False
        while start_idx <= len(times) - block_windows:
            block = np.arange(start_idx, start_idx + block_windows)
            if is_contiguous_block(dataset, block):
                blocks.append(block)
                target_time = times[start_idx] + month_stride_seconds
                found = True
                break
            start_idx += 1

        if not found:
            target_time += month_stride_seconds

    return blocks


def aggregate_spectrum(
    model: TransformerRegressor,
    dataset: HXMTWindowDataset,
    indices: Sequence[int],
    device: torch.device,
    target_log1p: bool,
) -> Dict[str, np.ndarray]:
    pred_windows = model_window_pass(model, dataset, indices, device, target_log1p)
    obs_windows = observed_window_pass(dataset, indices)
    pred_spectrum = pred_windows.sum(axis=0)
    obs_spectrum = obs_windows.sum(axis=0)
    residual = obs_spectrum - pred_spectrum

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    total_obs = float(obs_spectrum.sum())
    total_pred = float(pred_spectrum.sum())
    total_rel = float((total_pred - total_obs) / max(total_obs, 1e-6))

    return {
        "pred_spectrum": pred_spectrum,
        "obs_spectrum": obs_spectrum,
        "residual": residual,
        "mae": mae,
        "rmse": rmse,
        "total_obs": total_obs,
        "total_pred": total_pred,
        "total_rel": total_rel,
    }


def plot_spectrum(
    obs: np.ndarray,
    pred: np.ndarray,
    residual: np.ndarray,
    channels: np.ndarray,
    output_path: Path,
    block_duration: int,
    eval_idx: int,
    segment_id: int,
    met_start: float,
    met_end: float,
) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_spec, ax_res = axes


    energy   = channels #*370/256 + 15
    #ax_spec.errorbar(channels, obs, np.sqrt(obs), label="Observed", color="#1f77b4", linewidth=1.2)
    #ax_spec.errorbar(channels, pred, label="Predicted", color="#ff7f0e", linewidth=1.2)
    ax_spec.errorbar(energy, obs, np.sqrt(obs), label="Observed", color="#1f77b4", linewidth=1.2)
    ax_spec.errorbar(energy, pred, label="Predicted", color="#ff7f0e", linewidth=1.2)
    ax_spec.set_ylabel("Counts (aggregated)")
    ax_spec.set_yscale("log")
    #ax_spec.set_xscale('log')
    ax_spec.legend()
    ax_spec.grid(alpha=0.25)
    ax_spec.set_title(
        f"Eval #{eval_idx} | Segment {segment_id} | MET {met_start:.0f}–{met_end:.0f} | {block_duration}s"
    )

    #ax_res.errorbar(channels, residual, np.sqrt(obs), color="#2ca02c", linewidth=1.0)
    ax_res.errorbar(energy, residual, np.sqrt(obs), color="#2ca02c", linewidth=1.0)
    ax_res.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax_res.set_xlabel("Channel")
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
    parser.add_argument(
        "--data-csv",
        type=Path,
        default=data_dir / "hxmt_merged_dataset_HE_v3.0.csv",
        help="Dataset used for evaluating the model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint,
        help="Checkpoint produced by train_transformer_hxmt_32s.py.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=32,
        help="Window length (seconds) used during training.",
    )
    parser.add_argument(
        "--block-duration",
        type=int,
        default=1024,
        help="Length of each evaluation block in seconds.",
    )
    parser.add_argument(
        "--month-stride-days",
        type=float,
        default=30.0,
        help="Approximate temporal stride between evaluation blocks (days).",
    )
    parser.add_argument(
        "--max-evals",
        type=int,
        default=12,
        help="Maximum number of evaluation blocks to generate.",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=10.0,
        help="Maximum MET gap defining a continuous observation segment.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional row cap for quick tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "artifacts",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional CSV to store evaluation metrics.",
    )
    parser.add_argument(
        "--min-ch",
        type=int,
        default=0,
        help="the lower boundary of channel",
    )
    parser.add_argument(
        "--max-ch",
        type=int,
        default=256,
        help="the upper boundary of channel",
    )



    parser.add_argument("--device", type=str, default=None, help="Override device (cuda|cpu|mps).")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.block_duration % args.window != 0:
        raise ValueError("--block-duration must be an integer multiple of --window.")

    device = torch.device(args.device) if args.device else auto_device()
    model, feature_cols, target_cols, mu, sigma, target_log1p = load_checkpoint(
        args.checkpoint, device
    )
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Using device: {device}")

    df = read_csv(args.data_csv.expanduser(), limit_rows=args.limit_rows)
    ensure_columns(df, list(feature_cols) + list(target_cols))

    dataset_stride = HXMTWindowDataset(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window=args.window,
        stride=args.window,
        max_gap=args.max_gap,
        mu=mu,
        sigma=sigma,
    )

    block_windows = args.block_duration // args.window
    month_stride_seconds = args.month_stride_days * 86400.0
    blocks = find_monthly_blocks(
        dataset_stride,
        block_windows=block_windows,
        month_stride_seconds=month_stride_seconds,
        max_evals=args.max_evals,
    )

    if not blocks:
        raise RuntimeError(
            "No evaluation blocks matching the requested configuration were found."
        )

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    channels = np.arange(len(target_cols))
    channels = np.arange(args.min_ch, args.max_ch+1, 1)
    summary: List[Dict[str, float]] = []

    for eval_idx, block in enumerate(blocks, start=1):
        spec = aggregate_spectrum(
            model,
            dataset_stride,
            block,
            device=device,
            target_log1p=target_log1p,
        )
        seg_id = int(dataset_stride.sample_segments[block[0]])
        met_start = float(dataset_stride.sample_met_start[block[0]])
        met_end = float(dataset_stride.sample_met_end[block[-1]])

        fig_path = output_dir / f"eval_images/spectrum_eval_{eval_idx:02d}_seg{seg_id}_met{met_start:.0f}.png"
        plot_spectrum(
            spec["obs_spectrum"],
            spec["pred_spectrum"],
            spec["residual"],
            channels,
            fig_path,
            args.block_duration,
            eval_idx,
            seg_id,
            met_start,
            met_end,
        )
        print(
            f"Eval {eval_idx:02d}: segment={seg_id} MET={met_start:.0f}-{met_end:.0f} "
            f"total_obs={spec['total_obs']:.1f} total_pred={spec['total_pred']:.1f} "
            f"mae={spec['mae']:.3f} rmse={spec['rmse']:.3f}"
        )

        summary.append(
            {
                "eval_idx": eval_idx,
                "segment": seg_id,
                "met_start": met_start,
                "met_end": met_end,
                "total_obs": spec["total_obs"],
                "total_pred": spec["total_pred"],
                "total_rel": spec["total_rel"],
                "mae": spec["mae"],
                "rmse": spec["rmse"],
            }
        )

    if args.metrics_csv:
        metrics_path = args.metrics_csv.expanduser()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(summary).to_csv(metrics_path, index=False)
        print(f"Saved evaluation summary to {metrics_path}")


if __name__ == "__main__":
    main()
