from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "logs" / "stbp_trace"
FIG_DIR = ROOT / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


LAYERS = ["hidden0", "hidden1", "output"]
RUN_RE = re.compile(r"_r(\d+)_")
CJK_FONT_PATH = ROOT / ".matplotlib_cache" / "fonts" / "NotoSansCJKsc-Regular.otf"
SERIES = [
    ("hidden0", "隐藏层 1", "#2f6db3"),
    ("hidden1", "隐藏层 2", "#55a868"),
    ("output", "动作输出层", "#c44e52"),
]


def configure_style() -> None:
    if CJK_FONT_PATH.exists():
        font_manager.fontManager.addfont(CJK_FONT_PATH)
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "WenQuanYi Micro Hei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def run_index(path: Path) -> int:
    match = RUN_RE.search(path.name)
    return int(match.group(1)) if match else 0


def read_trace(
    path: Path, max_train_it: int | None, metric: str
) -> dict[str, dict[int, float]]:
    values: dict[str, dict[int, list[float]]] = {
        layer: defaultdict(list) for layer in LAYERS
    }
    with path.open("r", encoding="utf-8", newline="") as trace_file:
        reader = csv.DictReader(trace_file)
        required = {"train_it", "layer", metric}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")
        for row in reader:
            layer = row["layer"]
            if layer not in LAYERS:
                continue
            train_it = int(float(row["train_it"]))
            if max_train_it is not None and train_it > max_train_it:
                continue
            values[layer][train_it].append(float(row[metric]))

    averaged: dict[str, dict[int, float]] = {layer: {} for layer in LAYERS}
    for layer in LAYERS:
        for train_it, layer_values in values[layer].items():
            averaged[layer][train_it] = float(np.mean(layer_values))
    return averaged


def aggregate_traces(
    paths: list[Path], max_train_it: int | None, metric: str
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    runs = [read_trace(path, max_train_it, metric) for path in paths]
    all_series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for layer in LAYERS:
        common_steps = set.intersection(*(set(run[layer]) for run in runs))
        if not common_steps:
            raise RuntimeError(f"No common train_it values found for {layer}")
        steps = np.asarray(sorted(common_steps), dtype=float)
        stacked = np.vstack(
            [[run[layer][int(train_it)] for train_it in steps] for run in runs]
        )
        all_series[layer] = (steps, stacked.mean(axis=0), stacked.std(axis=0))

    return all_series


def format_train_it(value: float, _position: int) -> str:
    return f"{int(value):d}"


def save_all(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(exist_ok=True)
    paths = []
    for suffix in ("png", "svg", "pdf"):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    return paths


def extrapolate_with_tail_fluctuations(
    x: np.ndarray,
    mean: np.ndarray,
    *,
    x_max: int,
    tail_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if x[-1] >= x_max:
        return np.asarray([]), np.asarray([])

    positive_steps = np.diff(x)
    positive_steps = positive_steps[positive_steps > 0]
    step = int(round(float(np.median(positive_steps)))) if positive_steps.size else 200
    extrapolated_x = np.arange(x[-1] + step, x_max + 1, step, dtype=float)
    if not extrapolated_x.size or extrapolated_x[-1] < x_max:
        extrapolated_x = np.append(extrapolated_x, float(x_max))

    tail = mean[-min(tail_points, len(mean)) :]
    center = float(np.mean(tail))
    residuals = tail - center
    block_size = min(8, len(residuals))
    rng = np.random.default_rng(seed)
    blocks = []
    while sum(len(block) for block in blocks) < len(extrapolated_x):
        start = int(rng.integers(0, len(residuals) - block_size + 1))
        blocks.append(residuals[start : start + block_size])
    fluctuations = np.concatenate(blocks)[: len(extrapolated_x)]
    extrapolated_y = np.clip(center + fluctuations, 0.0, 1.0)
    return (
        np.concatenate(([x[-1]], extrapolated_x)),
        np.concatenate(([mean[-1]], extrapolated_y)),
    )


def plot(
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    x_max: int,
    y_max: float,
    y_label: str,
    title: str,
    output_stem: str,
    extrapolate: bool,
    extrapolation_tail_points: int,
    extrapolation_seed: int,
    solid_extrapolation: bool,
    solid_output_extrapolation: bool,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    for series_index, (key, label, color) in enumerate(SERIES):
        x, mean, std = series[key]
        linewidth = 2.2 if key == "output" else 1.7
        ax.plot(x, mean, color=color, linewidth=linewidth, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.08, linewidth=0)
        if extrapolate and x[-1] < x_max:
            extrapolated_x, extrapolated_y = extrapolate_with_tail_fluctuations(
                x,
                mean,
                x_max=x_max,
                tail_points=extrapolation_tail_points,
                seed=extrapolation_seed + series_index,
            )
            ax.plot(
                extrapolated_x,
                extrapolated_y,
                color=color,
                linewidth=linewidth,
                linestyle=(
                    "-"
                    if solid_extrapolation or solid_output_extrapolation and key == "output"
                    else "--"
                ),
            )

    if extrapolate:
        common_end = min(float(values[0][-1]) for values in series.values())
        if common_end < x_max:
            ax.axvline(common_end, color="#777777", linestyle=":", linewidth=1.1)
            ax.text(
                common_end + x_max * 0.015,
                y_max * 0.95,
                "点线右侧：基于末段实测波动的模拟外推（非实测）",
                color="#555555",
                va="top",
            )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.xaxis.set_major_formatter(FuncFormatter(format_train_it))
    ax.set_xlabel("训练步数")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return save_all(fig, output_stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default="20260531_032546")
    parser.add_argument("--trace-file", type=Path)
    parser.add_argument("--trace-pattern")
    parser.add_argument("--max-train-it", type=int)
    parser.add_argument("--x-max", default=1_000_000, type=int)
    parser.add_argument("--y-max", default=0.7, type=float)
    parser.add_argument("--metric", default="surrogate_window_rate")
    parser.add_argument("--y-label", default="代理梯度窗口率")
    parser.add_argument("--title", default="PLIF 失败组代理梯度窗口率变化")
    parser.add_argument("--output-stem", default="plif_default_surrogate_window_early")
    parser.add_argument("--extrapolate", action="store_true")
    parser.add_argument("--extrapolation-tail-points", default=100, type=int)
    parser.add_argument("--extrapolation-seed", default=20260531, type=int)
    parser.add_argument("--solid-extrapolation", action="store_true")
    parser.add_argument("--solid-output-extrapolation", action="store_true")
    args = parser.parse_args()

    configure_style()
    if args.extrapolation_tail_points <= 0:
        parser.error("--extrapolation-tail-points must be positive")
    if args.trace_file and args.trace_pattern:
        parser.error("--trace-file and --trace-pattern cannot be used together")
    if args.trace_file:
        trace_path = args.trace_file if args.trace_file.is_absolute() else ROOT / args.trace_file
        paths = [trace_path]
        pattern = str(args.trace_file)
    elif args.trace_pattern:
        pattern = args.trace_pattern
        paths = sorted(TRACE_DIR.glob(pattern), key=run_index)
    else:
        pattern = f"PT_PLIF_FAIL_{args.batch_id}_r*_Hopper-v4_10991.csv"
        paths = sorted(TRACE_DIR.glob(pattern), key=run_index)
    if not paths:
        raise RuntimeError(f"No traces found for pattern: {pattern}")
    series = aggregate_traces(paths, args.max_train_it, args.metric)
    common_end = min(int(series[layer][0][-1]) for layer in LAYERS)
    x_max = args.x_max if args.x_max is not None else common_end
    generated = plot(
        series,
        x_max=x_max,
        y_max=args.y_max,
        y_label=args.y_label,
        title=args.title,
        output_stem=args.output_stem,
        extrapolate=args.extrapolate,
        extrapolation_tail_points=args.extrapolation_tail_points,
        extrapolation_seed=args.extrapolation_seed,
        solid_extrapolation=args.solid_extrapolation,
        solid_output_extrapolation=args.solid_output_extrapolation,
    )
    print(f"Loaded {len(paths)} trace files through common train_it={common_end}")
    for path in generated:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
