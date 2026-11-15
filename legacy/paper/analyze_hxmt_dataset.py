#!/usr/bin/env python3
"""
Analyze the HXMT merged dataset and produce summary statistics and figures.

The script prints a short textual summary, writes a Markdown report, and saves
plots for the SAT_X/SAT_Y observation distribution and the distributions of
COR and ELV. It also maps the satellite longitude/latitude coverage.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (set backend before importing)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = (SCRIPT_DIR / "../../data/hxmt_merged_dataset_HE_v3.0.csv").resolve()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

KEY_COLUMNS: Sequence[str] = (
    "SAT_X",
    "SAT_Y",
    "SAT_Z",
    "SAT_ALT",
    "SAT_LON",
    "SAT_LAT",
    "ELV",
    "DYE_ELV",
    "ANG_DIST",
    "COR",
    "T_SAA",
    "TN_SAA",
    "SUN_ANG",
    "MOON_ANG",
    "SUNSHINE",
)
HISTOGRAM_COLUMNS: Sequence[str] = ("COR", "ELV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summary statistics and figures for the HXMT dataset."
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the merged HXMT dataset CSV file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where reports and figures will be written.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional limit for the number of rows read from the CSV file.",
    )
    return parser.parse_args()


def load_dataset(csv_path: Path, limit_rows: int | None) -> pd.DataFrame:
    read_kwargs = {"low_memory": False}
    if limit_rows:
        read_kwargs["nrows"] = limit_rows
    dataframe = pd.read_csv(csv_path, **read_kwargs)
    return dataframe


def format_stat(value: float) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.3g}"


def build_column_report(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    lines: List[str] = [
        "## Key Column Coverage and Statistics",
        "",
        "| Column | Coverage (%) | Min | 5th pct | Median | 95th pct | Max | Mean | Std |",
        "| :----- | -----------: | --: | ------: | -----: | -------: | --: | ---: | --: |",
    ]
    missing_columns: List[str] = []

    total_rows = len(df)
    for column in columns:
        if column not in df.columns:
            missing_columns.append(column)
            continue

        series = df[column]
        non_null = series.notna().sum()
        coverage = (non_null / total_rows * 100) if total_rows else 0.0
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        if numeric_series.empty:
            stats = ["NA"] * 7
        else:
            describe = numeric_series.describe(percentiles=[0.05, 0.5, 0.95])
            stats = [
                format_stat(describe.get("min")),
                format_stat(describe.get("5%")),
                format_stat(describe.get("50%")),
                format_stat(describe.get("95%")),
                format_stat(describe.get("max")),
                format_stat(describe.get("mean")),
                format_stat(describe.get("std")),
            ]
        row = f"| {column} | {coverage:9.2f} | " + " | ".join(stats) + " |"
        lines.append(row)

    if missing_columns:
        lines.extend(
            [
                "",
                "Columns absent from the dataset:",
                ", ".join(sorted(missing_columns)),
            ]
        )
    return lines


def plot_satellite_xy(df: pd.DataFrame, output_dir: Path) -> Path | None:
    if "SAT_X" not in df.columns or "SAT_Y" not in df.columns:
        return None
    xy_data = df[["SAT_X", "SAT_Y"]].apply(pd.to_numeric, errors="coerce").dropna()
    if xy_data.empty:
        return None

    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(
        xy_data["SAT_X"],
        xy_data["SAT_Y"],
        gridsize=80,
        cmap="viridis",
        mincnt=1,
    )
    ax.set_title("HXMT Observation Distribution (SAT_X vs SAT_Y)")
    ax.set_xlabel("SAT_X")
    ax.set_ylabel("SAT_Y")
    colorbar = fig.colorbar(hb, ax=ax)
    colorbar.set_label("Counts")
    fig.tight_layout()

    output_path = output_dir / "sat_xy_distribution.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_histograms(
    df: pd.DataFrame, columns: Sequence[str], output_dir: Path
) -> Path | None:
    available_columns = [
        column for column in columns if column in df.columns
    ]
    if not available_columns:
        return None

    numeric_columns = {
        column: pd.to_numeric(df[column], errors="coerce").dropna()
        for column in available_columns
    }
    numeric_columns = {
        column: series for column, series in numeric_columns.items() if not series.empty
    }
    if not numeric_columns:
        return None

    n_panels = len(numeric_columns)
    n_cols = min(2, n_panels)
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    for ax, (column, series) in zip(axes_flat, numeric_columns.items()):
        ax.hist(series, bins=60, color="#1f77b4", alpha=0.75)
        ax.set_title(f"{column} Distribution")
        ax.set_xlabel(column)
        ax.set_ylabel("Counts")
        ax.grid(alpha=0.2)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle("Environmental Parameter Distributions")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = output_dir / "environmental_distributions.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_satellite_lonlat(df: pd.DataFrame, output_dir: Path) -> Path | None:
    lon_lat_columns = ["SAT_LON", "SAT_LAT"]
    if any(column not in df.columns for column in lon_lat_columns):
        return None

    coords = df[lon_lat_columns].apply(pd.to_numeric, errors="coerce").dropna()
    if coords.empty:
        return None

    # Normalize longitude to [-180, 180] to avoid wrap artifacts.
    longitudes = ((coords["SAT_LON"] + 180) % 360) - 180
    latitudes = coords["SAT_LAT"].clip(-90, 90)

    try:
        import cartopy.crs as ccrs  # type: ignore
        import cartopy.feature as cfeature  # type: ignore
    except Exception as exc:
        print(f"[warn] Cartopy not available ({exc}); using fallback lon/lat scatter plot.")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        hb = ax.hexbin(longitudes, latitudes, gridsize=120, cmap="plasma", mincnt=1)
        ax.set_title("HXMT Satellite Ground Track Coverage (Lon/Lat)")
        ax.set_xlabel("Longitude (deg)")
        ax.set_ylabel("Latitude (deg)")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        colorbar = fig.colorbar(hb, ax=ax)
        colorbar.set_label("Counts")
    else:
        projection = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1, projection=projection)
        ax.set_title("HXMT Satellite Ground Track Coverage")
        ax.set_global()
        ax.add_feature(cfeature.LAND.with_scale("110m"), facecolor="lightgray")
        ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor="white")
        ax.add_feature(cfeature.COASTLINE.with_scale("110m"), linewidth=0.5)
        ax.add_feature(cfeature.BORDERS.with_scale("110m"), linewidth=0.3)
        ax.gridlines(draw_labels=True, linewidth=0.2, color="gray", alpha=0.5, linestyle="--")
        hb = ax.hexbin(
            longitudes,
            latitudes,
            gridsize=160,
            transform=projection,
            cmap="plasma",
            mincnt=1,
            zorder=3,
        )
        colorbar = fig.colorbar(hb, ax=ax, orientation="vertical", pad=0.02, shrink=0.8)
        colorbar.set_label("Counts")

    fig.tight_layout()
    output_path = output_dir / "sat_lonlat_coverage.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def write_report(
    report_path: Path,
    dataset_path: Path,
    dataframe: pd.DataFrame,
    report_lines: Sequence[str],
    sat_xy_path: Path | None,
    hist_path: Path | None,
    lonlat_path: Path | None,
) -> None:
    total_rows, total_columns = dataframe.shape
    header = [
        "# HXMT Dataset Summary",
        "",
        f"- Source file: `{dataset_path}`",
        f"- Rows loaded: {total_rows}",
        f"- Columns: {total_columns}",
    ]
    if "MET" in dataframe.columns:
        met_numeric = pd.to_numeric(dataframe["MET"], errors="coerce").dropna()
        if not met_numeric.empty:
            header.append(
                f"- MET span: {met_numeric.min():.3f} — {met_numeric.max():.3f} "
                f"(Δ={met_numeric.max() - met_numeric.min():.3f})"
            )

    figure_lines = []
    if sat_xy_path:
        figure_lines.append(f"- SAT_X/SAT_Y distribution: `{sat_xy_path}`")
    if hist_path:
        figure_lines.append(f"- Environmental distributions: `{hist_path}`")
    if lonlat_path:
        figure_lines.append(f"- SAT_LON/SAT_LAT coverage: `{lonlat_path}`")
    if figure_lines:
        header.extend(["", "Generated figures:"] + figure_lines)

    content = "\n".join([*header, "", *report_lines, ""])
    report_path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).expanduser().resolve()
    if not dataset_path.exists():
        print(f"[error] Dataset not found at: {dataset_path}")
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] Loading dataset from {dataset_path} ...")
    dataframe = load_dataset(dataset_path, args.limit_rows)
    print(
        f"[info] Loaded {len(dataframe):,} rows and {dataframe.shape[1]} columns."
    )

    report_lines = build_column_report(dataframe, KEY_COLUMNS)

    print("[info] Generating figures ...")
    sat_xy_path = plot_satellite_xy(dataframe, output_dir)
    hist_path = plot_histograms(dataframe, HISTOGRAM_COLUMNS, output_dir)
    lonlat_path = plot_satellite_lonlat(dataframe, output_dir)

    report_path = output_dir / "hxmt_dataset_report.md"
    write_report(
        report_path,
        dataset_path,
        dataframe,
        report_lines,
        sat_xy_path,
        hist_path,
        lonlat_path,
    )
    print(f"[info] Report written to {report_path}")
    if sat_xy_path:
        print(f"[info] SAT_X/SAT_Y figure: {sat_xy_path}")
    if hist_path:
        print(f"[info] COR/ELV distributions figure: {hist_path}")
    if lonlat_path:
        print(f"[info] SAT_LON/SAT_LAT coverage figure: {lonlat_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
