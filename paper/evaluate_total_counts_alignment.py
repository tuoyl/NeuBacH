#!/usr/bin/env python3
"""
Evaluate the HXMT transformer background model by comparing observed and
predicted count rates for every inference window. The script loads the trained
checkpoint, reproduces the windowed dataset, and produces a scatter plot
(observed vs. predicted rates) together with summary metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (set backend before importing)


SCRIPT_DIR = Path(__file__).resolve().parent

# Ensure the repository root (parent of `paper/`) is on the Python path so we can
# reuse training utilities without duplicating logic.
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_transformer_hxmt_32s import (  # noqa: E402
    HXMTWindowDataset,
    TransformerRegressor,
    auto_device,
    evaluate_dataset,
    read_csv,
    set_seed,
)


def resolve_first_existing(base: Path, candidates: Sequence[str]) -> Path:
    for candidate in candidates:
        path = (base / candidate).resolve()
        if path.exists():
            return path
    return (base / candidates[0]).resolve()


def default_dataset_path() -> Path:
    candidates = (
        "../../data/hxmt_merged_dataset_HE_v3.0.csv",
        "../../../data/hxmt_merged_dataset_HE_v3.0.csv",
        "../hxmt_merged_dataset_HE_v3.0.csv",
    )
    return resolve_first_existing(SCRIPT_DIR, candidates)


def default_checkpoint_path() -> Path:
    candidates = (
        "../../script/artifacts/best_model.pt",
        "../best_model.pt",
        "../../../script/artifacts/best_model.pt",
    )
    return resolve_first_existing(SCRIPT_DIR, candidates)


def load_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Tuple[
    TransformerRegressor,
    Sequence[str],
    Sequence[str],
    np.ndarray,
    np.ndarray,
    bool,
    dict,
]:
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

    return model, feature_cols, target_cols, mu, sigma, target_log1p, model_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset_path(),
        help="Path to the merged HXMT dataset CSV.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=default_checkpoint_path(),
        help="Trained model checkpoint (best_model.pt).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Window length (seconds). Defaults to value stored in the checkpoint.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride (seconds) between windows. Defaults to checkpoint value.",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=None,
        help="Maximum MET gap for continuous segments. Defaults to checkpoint value.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size during inference.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional row cap for quick dry-runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu|cuda|mps). Defaults to automatic selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "outputs",
        help="Directory for plots and generated artifacts.",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="count_rate_alignment.png",
        help="Filename for the scatter plot.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Observed vs. Predicted Count Rates",
        help="Custom title for the scatter plot.",
    )
    return parser.parse_args()


def compute_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    observed_counts_total: float | None = None,
    predicted_counts_total: float | None = None,
) -> dict:
    residual = predicted - observed
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    rel = np.abs(observed)
    mape = float(np.mean(np.abs(residual) / np.maximum(rel, 1e-6)) * 100.0)
    corr = float(np.corrcoef(observed, predicted)[0, 1]) if observed.size > 1 else float("nan")

    # Linear regression y = a * x + b
    A = np.vstack([observed, np.ones_like(observed)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, predicted, rcond=None)[0]
    except np.linalg.LinAlgError:
        slope, intercept = float("nan"), float("nan")

    mean_obs = float(np.mean(observed))
    mean_pred = float(np.mean(predicted))

    if observed_counts_total is None:
        total_obs = float(observed.sum())
    else:
        total_obs = float(observed_counts_total)

    if predicted_counts_total is None:
        total_pred = float(predicted.sum())
    else:
        total_pred = float(predicted_counts_total)

    total_rel_err = (total_pred - total_obs) / max(total_obs, 1e-6)

    return {
        "mae": mae,
        "rmse": rmse,
        "mape_percent": mape,
        "corr": corr,
        "slope": slope,
        "intercept": intercept,
        "mean_observed": mean_obs,
        "mean_predicted": mean_pred,
        "total_observed": total_obs,
        "total_predicted": total_pred,
        "total_relative_error": total_rel_err,
    }


def plot_alignment(
    observed: np.ndarray,
    predicted: np.ndarray,
    metrics: dict,
    output_path: Path,
    title: str,
) -> None:
    min_val = float(np.min([observed.min(), predicted.min()]))
    max_val = float(np.max([observed.max(), predicted.max()]))
    padding = 0.02 * (max_val - min_val) if max_val > min_val else 1.0
    line_min = min_val - padding
    line_max = max_val + padding

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(
        observed,
        predicted,
        s=14,
        alpha=1.0,
        edgecolors="none",
        color="#1f77b4",
        label="Windows",
    )
    ax.errorbar(
        observed,
        predicted,
        xerr=np.sqrt(observed),
        fmt='.',
        lw=0.4,
        alpha=0.4,
        capsize=4,
    )
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        color="#ff7f0e",
        linestyle="--",
        linewidth=1.2,
        label="Ideal (y = x)",
    )

    subtitle = (
        f"MAE={metrics['mae']:.2f} | RMSE={metrics['rmse']:.2f} | "
        f"MAPE={metrics['mape_percent']:.2f}% | r={metrics['corr']:.4f}\n"
        f"Mean rate obs={metrics['mean_observed']:.2e}, pred={metrics['mean_predicted']:.2e} "
        f"| Total counts Δ={metrics['total_relative_error']*100:.2f}%"
    )

    ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    ax.set_xlabel("Observed count rate per window (counts/sec)")
    ax.set_ylabel("Predicted count rate per window (counts/sec)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else auto_device()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    dataset_path = args.dataset.expanduser().resolve()

    print(f"[info] Using device: {device}")
    print(f"[info] Loading checkpoint: {checkpoint_path}")
    model, feature_cols, target_cols, mu, sigma, target_log1p, model_args = load_checkpoint(
        checkpoint_path, device
    )

    window = args.window or int(model_args.get("window", 32))
    stride = args.stride or int(model_args.get("stride", window))
    max_gap = args.max_gap or float(model_args.get("max_gap", 10.0))

    print(f"[info] Dataset: {dataset_path}")
    df = read_csv(dataset_path, limit_rows=args.limit_rows)
    print(f"[info] Loaded {len(df):,} rows with {df.shape[1]} columns.")

    dataset = HXMTWindowDataset(
        df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        window=window,
        stride=stride,
        max_gap=max_gap,
        mu=mu,
        sigma=sigma,
    )
    print(f"[info] Prepared {len(dataset):,} windows (window={window}s stride={stride}s).")

    preds, targets = evaluate_dataset(
        model,
        dataset,
        device=device,
        use_log_target=target_log1p,
        batch_size=args.batch_size,
    )
    if targets is None:
        raise RuntimeError("Targets array is empty; unable to compute alignment plot.")

    observed_totals = targets.sum(axis=1)
    predicted_totals = preds.sum(axis=1)

    window_duration = float(window)
    observed_rates = observed_totals / window_duration
    predicted_rates = predicted_totals / window_duration

    total_observed_counts = float(observed_totals.sum())
    total_predicted_counts = float(predicted_totals.sum())

    metrics = compute_metrics(
        observed_rates,
        predicted_rates,
        observed_counts_total=total_observed_counts,
        predicted_counts_total=total_predicted_counts,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_path = output_dir / args.figure_name
    plot_alignment(observed_rates, predicted_rates, metrics, output_path, args.title)

    print("[info] Evaluation complete.")
    print(f"[info] Figure saved to: {output_path}")
    print(
        "[info] count totals | observed={:.4e} predicted={:.4e} relative_error={:+.3f}%".format(
            metrics["total_observed"], metrics["total_predicted"], metrics["total_relative_error"] * 100.0
        )
    )
    print(
        "[info] metrics | mae={:.3f} rmse={:.3f} mape={:.3f}% corr={:.5f} slope={:.5f} intercept={:.3f}".format(
            metrics["mae"],
            metrics["rmse"],
            metrics["mape_percent"],
            metrics["corr"],
            metrics["slope"],
            metrics["intercept"],
        )
    )
    print(
        "[info] rates | mean_observed={:.4e} mean_predicted={:.4e} counts_window={}s".format(
            metrics["mean_observed"], metrics["mean_predicted"], window
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
