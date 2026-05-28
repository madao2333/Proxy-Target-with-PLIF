from __future__ import annotations

import csv
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "logs" / "stbp_trace"
FIG_DIR = ROOT / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUN_RE = re.compile(r"_r(\d+)_")
LAYERS = ["hidden0", "hidden1", "output"]
ZERO_FLOOR = 1e-12
X_LABEL = "训练步数（百万）"
LAYER_LABELS = {
    "hidden0": "第一隐藏层",
    "hidden1": "第二隐藏层",
    "output": "动作输出层",
}


@dataclass(frozen=True)
class TraceGroup:
    key: str
    label: str
    pattern: str
    color: str
    linestyle: str


GROUPS = [
    TraceGroup(
        key="fail",
        label="默认失败组（2 次评价网络更新后更新 1 次策略网络）",
        pattern="PT_PLIF_FAIL_20260513_003231_r*_Hopper-v4_10991.csv",
        color="#c44e52",
        linestyle="--",
    ),
    TraceGroup(
        key="policy_freq",
        label="调整组（4 次评价网络更新后更新 1 次策略网络）",
        pattern="PT_PLIF_POLICY_FREQ_20260513_215542_r*_Hopper-v4_10991.csv",
        color="#2f6db3",
        linestyle="-",
    ),
]


METRICS = [
    "pre_spike_rate",
    "post_spike_rate",
    "current_abs_mean",
    "volt_mean",
    "volt_std",
    "current_grad_abs_mean",
    "weight_grad_t_l2",
    "param_weight_grad_l2",
    "plif_tau_grad_t_abs",
]

GRADIENT_METRICS = [
    ("current_grad_abs_mean", r"$\left|\partial\mathcal{L}/\partial I_t\right|$"),
    ("weight_grad_t_l2", r"$\|\nabla_{W,t}\mathcal{L}\|_2$"),
    ("param_weight_grad_l2", r"$\|\nabla_W\mathcal{L}\|_2$"),
    ("plif_tau_grad_t_abs", r"$\left|\partial\mathcal{L}/\partial w_l\right|$"),
]

SPIKE_METRICS = [
    ("pre_spike_rate", r"$\mathbb{E}[s_t^{l-1}]$"),
    ("post_spike_rate", r"$\mathbb{E}[s_t^l]$"),
]

STATE_METRICS = [
    ("current_abs_mean", r"$\mathbb{E}[|I_t|]$"),
    ("volt_mean", r"$\mathbb{E}[v_t]$"),
    ("volt_std", r"$\mathrm{Std}(v_t)$"),
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


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def check_trace_schema(path: Path, required: set[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or [])
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {sorted(missing)}")


def read_trace_file(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    sums: dict[str, dict[int, dict[str, float]]] = {
        layer: defaultdict(lambda: defaultdict(float)) for layer in LAYERS
    }
    counts: dict[str, dict[int, dict[str, int]]] = {
        layer: defaultdict(lambda: defaultdict(int)) for layer in LAYERS
    }

    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            layer = row.get("layer", "")
            if layer not in LAYERS:
                continue
            train_it_value = parse_float(row.get("train_it"))
            if train_it_value is None:
                continue
            train_it = int(train_it_value)
            for metric in METRICS:
                source_metric = "plif_tau_grad_t" if metric == "plif_tau_grad_t_abs" else metric
                value = parse_float(row.get(source_metric))
                if value is None:
                    continue
                if metric.endswith("_abs"):
                    value = abs(value)
                sums[layer][train_it][metric] += value
                counts[layer][train_it][metric] += 1

    averaged: dict[str, dict[int, dict[str, float]]] = {layer: {} for layer in LAYERS}
    for layer in LAYERS:
        for train_it, metric_sums in sums[layer].items():
            averaged[layer][train_it] = {}
            for metric, value_sum in metric_sums.items():
                averaged[layer][train_it][metric] = value_sum / counts[layer][train_it][metric]
    return averaged


def read_all_traces() -> dict[str, dict[int, dict[str, dict[int, dict[str, float]]]]]:
    required = {
        "train_it",
        "layer",
        "pre_spike_rate",
        "post_spike_rate",
        "current_abs_mean",
        "volt_mean",
        "volt_std",
        "current_grad_abs_mean",
        "weight_grad_t_l2",
        "param_weight_grad_l2",
        "plif_tau_grad_t",
    }
    data: dict[str, dict[int, dict[str, dict[int, dict[str, float]]]]] = {}
    for group in GROUPS:
        paths = sorted(TRACE_DIR.glob(group.pattern), key=run_index)
        if len(paths) != 5:
            names = "\n".join(str(path.relative_to(ROOT)) for path in paths)
            raise RuntimeError(f"Expected 5 traces for {group.key}, found {len(paths)}:\n{names}")
        for path in paths:
            check_trace_schema(path, required)
        data[group.key] = {run_index(path): read_trace_file(path) for path in paths}
    return data


def aggregate_series(
    group_runs: dict[int, dict[str, dict[int, dict[str, float]]]],
    layer: str,
    metric: str,
    *,
    floor_zero: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_steps: set[int] | None = None
    for run_data in group_runs.values():
        steps = {step for step, values in run_data[layer].items() if metric in values}
        common_steps = steps if common_steps is None else common_steps.intersection(steps)
    if not common_steps:
        raise RuntimeError(f"No common steps for {layer}/{metric}")

    steps_array = np.asarray(sorted(common_steps), dtype=float)
    run_values = []
    for run_data in group_runs.values():
        values = np.asarray([run_data[layer][int(step)][metric] for step in steps_array], dtype=float)
        if floor_zero:
            values = np.maximum(values, ZERO_FLOOR)
        run_values.append(values)
    stacked = np.vstack(run_values)
    return steps_array / 1_000_000.0, stacked.mean(axis=0), stacked.std(axis=0)


def save_all(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "svg", "pdf"):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_gradient_collapse(data: dict[str, dict[int, dict[str, dict[int, dict[str, float]]]]]) -> list[Path]:
    fig, axes = plt.subplots(len(LAYERS), len(GRADIENT_METRICS), figsize=(14.4, 8.8), sharex=True)
    for row, layer in enumerate(LAYERS):
        for col, (metric, title) in enumerate(GRADIENT_METRICS):
            ax = axes[row, col]
            for group in GROUPS:
                x, mean, std = aggregate_series(data[group.key], layer, metric, floor_zero=True)
                ax.plot(x, mean, color=group.color, linestyle=group.linestyle, linewidth=1.9, label=group.label)
                lower = np.maximum(mean - std, ZERO_FLOOR)
                upper = np.maximum(mean + std, ZERO_FLOOR)
                ax.fill_between(x, lower, upper, color=group.color, alpha=0.10, linewidth=0)
            ax.set_yscale("log")
            ax.set_title(title)
            if col == 0:
                ax.set_ylabel(LAYER_LABELS[layer])
            if row == len(LAYERS) - 1:
                ax.set_xlabel(X_LABEL)
            ax.margins(x=0.01)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("分层反向梯度对比", y=0.995, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.045, f"对数坐标中的零值显示为 {ZERO_FLOOR:g}", ha="center")
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    return save_all(fig, "plif_stbp_gradient_collapse")


def plot_spike_rates(data: dict[str, dict[int, dict[str, dict[int, dict[str, float]]]]]) -> list[Path]:
    fig, axes = plt.subplots(len(LAYERS), len(SPIKE_METRICS), figsize=(11.6, 8.4), sharex=True, sharey=True)
    for row, layer in enumerate(LAYERS):
        for col, (metric, title) in enumerate(SPIKE_METRICS):
            ax = axes[row, col]
            for group in GROUPS:
                x, mean, std = aggregate_series(data[group.key], layer, metric)
                ax.plot(x, mean, color=group.color, linestyle=group.linestyle, linewidth=1.9, label=group.label)
                ax.fill_between(x, mean - std, mean + std, color=group.color, alpha=0.10, linewidth=0)
            ax.set_ylim(0, 0.8)
            ax.set_title(title)
            if col == 0:
                ax.set_ylabel(LAYER_LABELS[layer])
            if row == len(LAYERS) - 1:
                ax.set_xlabel(X_LABEL)
            ax.margins(x=0.01)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("阈值前后放电率对比", y=0.995, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    return save_all(fig, "plif_stbp_spike_rates")


def plot_state_diagnostics(data: dict[str, dict[int, dict[str, dict[int, dict[str, float]]]]]) -> list[Path]:
    fig, axes = plt.subplots(len(LAYERS), len(STATE_METRICS), figsize=(13.2, 8.6), sharex=True)
    for row, layer in enumerate(LAYERS):
        for col, (metric, title) in enumerate(STATE_METRICS):
            ax = axes[row, col]
            if metric == "volt_mean":
                ax.axhline(0.5, color="#666666", linestyle=":", linewidth=1.0, label=r"$V_{\mathrm{th}}=0.5$")
            for group in GROUPS:
                x, mean, std = aggregate_series(data[group.key], layer, metric)
                ax.plot(x, mean, color=group.color, linestyle=group.linestyle, linewidth=1.9, label=group.label)
                ax.fill_between(x, mean - std, mean + std, color=group.color, alpha=0.10, linewidth=0)
            ax.set_title(title)
            if col == 0:
                ax.set_ylabel(LAYER_LABELS[layer])
            if row == len(LAYERS) - 1:
                ax.set_xlabel(X_LABEL)
            ax.margins(x=0.01)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    fig.legend(dedup.values(), dedup.keys(), loc="lower center", ncol=3)
    fig.suptitle("电流与膜电位统计", y=0.995, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.065, 1, 0.965))
    return save_all(fig, "plif_stbp_state_diagnostics")


def max_gradient_for_run(run_data: dict[str, dict[int, dict[str, float]]], step: int) -> float:
    metrics = ["current_grad_abs_mean", "weight_grad_t_l2", "param_weight_grad_l2", "plif_tau_grad_t_abs"]
    maximum = 0.0
    for layer in LAYERS:
        values = run_data[layer].get(step, {})
        for metric in metrics:
            maximum = max(maximum, abs(values.get(metric, 0.0)))
    return maximum


def print_summary(data: dict[str, dict[int, dict[str, dict[int, dict[str, float]]]]]) -> None:
    for group in GROUPS:
        print(group.label)
        for run, run_data in sorted(data[group.key].items()):
            steps = sorted(set().union(*(set(run_data[layer]) for layer in LAYERS)))
            nonzero_steps = [step for step in steps if max_gradient_for_run(run_data, step) > 0.0]
            last_nonzero = nonzero_steps[-1] if nonzero_steps else None
            late_steps = [step for step in steps if step >= 900_000]
            late_max_grad = max((max_gradient_for_run(run_data, step) for step in late_steps), default=0.0)
            late_post_by_layer = []
            for layer in LAYERS:
                layer_values = [
                    run_data[layer][step]["post_spike_rate"]
                    for step in late_steps
                    if "post_spike_rate" in run_data[layer][step]
                ]
                late_post_by_layer.append(f"{layer}={np.mean(layer_values):.3f}")
            print(
                f"  run {run}: last nonzero gradient train_it={last_nonzero}, "
                f"late max gradient={late_max_grad:.3e}, late post-spike {' '.join(late_post_by_layer)}"
            )


def main() -> None:
    configure_style()
    data = read_all_traces()
    paths = []
    paths.extend(plot_gradient_collapse(data))
    paths.extend(plot_spike_rates(data))
    paths.extend(plot_state_diagnostics(data))
    print_summary(data)
    print("\nGenerated figures:")
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
