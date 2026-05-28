from __future__ import annotations

import csv
import math
import os
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "logs" / "stbp_trace"
FIG_DIR = ROOT / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ENVS = [
    "Ant-v4",
    "HalfCheetah-v4",
    "Hopper-v4",
    "InvertedDoublePendulum-v4",
    "Walker2d-v4",
]
ENV_LABELS = {
    "Ant-v4": "Ant",
    "HalfCheetah-v4": "HalfCheetah",
    "Hopper-v4": "Hopper",
    "InvertedDoublePendulum-v4": "InvDoublePendulum",
    "Walker2d-v4": "Walker2d",
}
ENV_COLORS = {
    "Ant-v4": "#4c78a8",
    "HalfCheetah-v4": "#f58518",
    "Hopper-v4": "#54a24b",
    "InvertedDoublePendulum-v4": "#b279a2",
    "Walker2d-v4": "#e45756",
}
LAYERS = ["hidden0", "hidden1", "output"]
LAYER_MARKERS = {"hidden0": "o", "hidden1": "s", "output": "^"}
LAYER_LABELS = {
    "hidden0": "第一隐藏层",
    "hidden1": "第二隐藏层",
    "output": "动作输出层",
}

LIF_PATTERN = "PT_LIF_DEFAULT_ALL_ENVS_20260516_105755_*.csv"
PLIF_PATTERN = "PT_PLIF_POLICY_FREQ_ALL_ENVS_20260515_141555_*.csv"
LATE_TRAIN_IT = 900_000

METRICS = [
    ("post_spike_rate", r"$\mathbb{E}[s_t^l]$", False),
    ("current_abs_mean", r"$\mathbb{E}[|I_t^l|]$", False),
    ("volt_std", r"$\mathrm{Std}(v_t^l)$", False),
    (
        "current_grad_abs_mean",
        r"$\mathbb{E}[|\partial \mathcal{L}/\partial I_t^l|]$",
        True,
    ),
    (
        "surrogate_window_rate",
        r"$\mathbb{E}[\mathbb{I}(|v_t^l-V_{\mathrm{th}}|<\Delta)]$",
        False,
    ),
    ("effective_retention_mean", r"$\mathbb{E}[\tau_l(1-s_{t-1}^l)]$", False),
    (
        "retained_voltage_abs_mean",
        r"$\mathbb{E}[|v_{t-1}^l\tau_l(1-s_{t-1}^l)|]$",
        False,
    ),
    (
        "current_grad_nonzero_rate",
        r"$\mathbb{E}[\mathbb{I}(\partial \mathcal{L}/\partial I_t^l\ne 0)]$",
        False,
    ),
    ("action_grad_l2", r"$\|\partial \mathcal{L}/\partial a\|_2$", True),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "WenQuanYi Micro Hei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
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


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def env_and_seed(path: Path) -> tuple[str, int]:
    env = next((candidate for candidate in ENVS if candidate in path.name), "")
    seed_match = re.search(r"_(\d+)\.csv$", path.name)
    if not env or seed_match is None:
        raise ValueError(f"Cannot parse environment/seed from {path.name}")
    return env, int(seed_match.group(1))


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def pearson(x_values: np.ndarray, y_values: np.ndarray) -> float:
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if mask.sum() < 3:
        return float("nan")
    x = x_values[mask]
    y = y_values[mask]
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def read_trace_means(pattern: str) -> dict[tuple[str, int, str], dict[str, float]]:
    metrics = [metric for metric, _, _ in METRICS]
    values: dict[tuple[str, int, str], dict[str, list[float]]] = {}

    paths = sorted(TRACE_DIR.glob(pattern))
    if not paths:
        raise RuntimeError(f"No trace files matched {pattern}")

    for path in paths:
        env, seed = env_and_seed(path)
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            missing = set(metrics + ["train_it", "layer"]).difference(reader.fieldnames or [])
            if missing:
                raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")
            for row in reader:
                train_it = parse_float(row.get("train_it"))
                layer = row.get("layer", "")
                if train_it is None or train_it < LATE_TRAIN_IT or layer not in LAYERS:
                    continue
                key = (env, seed, layer)
                layer_values = values.setdefault(key, {metric: [] for metric in metrics})
                for metric in metrics:
                    value = parse_float(row.get(metric))
                    if value is not None:
                        layer_values[metric].append(value)

    return {
        key: {metric: mean(metric_values) for metric, metric_values in layer_values.items()}
        for key, layer_values in values.items()
    }


def paired_points(
    lif: dict[tuple[str, int, str], dict[str, float]],
    plif: dict[tuple[str, int, str], dict[str, float]],
    metric: str,
) -> list[tuple[tuple[str, int, str], float, float]]:
    points = []
    for key in sorted(set(lif).intersection(plif)):
        x_value = lif[key].get(metric, float("nan"))
        y_value = plif[key].get(metric, float("nan"))
        if math.isfinite(x_value) and math.isfinite(y_value):
            points.append((key, x_value, y_value))
    return points


def set_matched_limits(ax: plt.Axes, x_values: np.ndarray, y_values: np.ndarray, log_scale: bool) -> None:
    combined = np.concatenate([x_values, y_values])
    combined = combined[np.isfinite(combined)]
    if log_scale:
        combined = combined[combined > 0]
        lower = float(combined.min() * 0.72)
        upper = float(combined.max() * 1.35)
        ax.set_xscale("log")
        ax.set_yscale("log")
    else:
        lower = float(combined.min())
        upper = float(combined.max())
        span = upper - lower
        padding = 0.08 * span if span > 0 else max(abs(upper) * 0.08, 0.05)
        lower -= padding
        upper += padding
        if lower >= 0 and upper <= 1.05:
            lower = max(0.0, lower)
            upper = min(1.05, upper)
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.plot([lower, upper], [lower, upper], color="#666666", linestyle=":", linewidth=1.0, zorder=1)


def save_all(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "svg", "pdf"):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_success_mechanism() -> list[Path]:
    lif = read_trace_means(LIF_PATTERN)
    plif = read_trace_means(PLIF_PATTERN)

    fig, axes = plt.subplots(3, 3, figsize=(12.4, 10.1))
    stats: list[tuple[str, float, int]] = []

    for ax, (metric, title, log_scale) in zip(axes.ravel(), METRICS):
        points = paired_points(lif, plif, metric)
        x_values = np.asarray([point[1] for point in points], dtype=float)
        y_values = np.asarray([point[2] for point in points], dtype=float)
        correlation = pearson(x_values, y_values)
        stats.append((metric, correlation, len(points)))

        set_matched_limits(ax, x_values, y_values, log_scale)
        for (env, _seed, layer), x_value, y_value in points:
            ax.scatter(
                x_value,
                y_value,
                s=33,
                marker=LAYER_MARKERS[layer],
                color=ENV_COLORS[env],
                edgecolor="white",
                linewidth=0.55,
                alpha=0.92,
                zorder=3,
            )

        ax.set_title(f"{title}\nr = {correlation:.3f}")
        ax.set_xlabel("LIF")
        ax.set_ylabel("PLIF")
        ax.set_aspect("equal", adjustable="box")

    env_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6,
            markerfacecolor=ENV_COLORS[env],
            markeredgecolor="white",
            label=ENV_LABELS[env],
        )
        for env in ENVS
    ]
    layer_handles = [
        Line2D(
            [0],
            [0],
            marker=LAYER_MARKERS[layer],
            linestyle="",
            markersize=6,
            markerfacecolor="#444444",
            markeredgecolor="white",
            label=LAYER_LABELS[layer],
        )
        for layer in LAYERS
    ]

    fig.legend(
        handles=env_handles,
        loc="lower center",
        bbox_to_anchor=(0.35, 0.006),
        ncol=5,
        title="环境",
    )
    fig.legend(
        handles=layer_handles,
        loc="lower center",
        bbox_to_anchor=(0.79, 0.006),
        ncol=3,
        title="网络层",
    )
    fig.suptitle(
        "成功训练组 LIF 与 PLIF 配对机制诊断",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))

    print("Late-window paired correlations (train_it >= 900000):")
    for metric, correlation, count in stats:
        print(f"  {metric}: r={correlation:.4f}, n={count}")

    return save_all(fig, "lif_plif_success_mechanism_paired")


def main() -> None:
    configure_style()
    paths = plot_success_mechanism()
    print("\nGenerated figures:")
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
