from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs" / "log_groups"
FIG_DIR = ROOT / "figures"
EVAL_FREQ = 5000

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EVAL_RE = re.compile(r"Evaluation over 10 episodes:\s*([-+]?\d+(?:\.\d+)?)")
RUN_RE = re.compile(r"_r(\d+)_")


@dataclass(frozen=True)
class RunGroup:
    key: str
    label: str
    log_group: str
    pattern: str
    color: str
    linestyle: str


GROUPS = [
    RunGroup(
        key="fail",
        label="默认失败组（policy_freq=2）",
        log_group="PT_PLIF_FAIL",
        pattern="PT_PLIF_FAIL_20260513_003231_r*_Hopper-v4_seed10991_gpu*.log",
        color="#c44e52",
        linestyle="--",
    ),
    RunGroup(
        key="policy_freq",
        label="更新间隔调整组（policy_freq=4）",
        log_group="PT_PLIF_POLICY_FREQ",
        pattern="PT_PLIF_POLICY_FREQ_20260513_215542_r*_Hopper-v4_seed10991_gpu*.log",
        color="#2f6db3",
        linestyle="-",
    ),
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
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
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


def parse_evaluations(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        values = [float(match.group(1)) for line in f for match in [EVAL_RE.search(line)] if match]
    if not values:
        raise ValueError(f"No evaluation records found in {path}")
    return values


def run_index(path: Path) -> int:
    match = RUN_RE.search(path.name)
    return int(match.group(1)) if match else 0


def read_group(group: RunGroup) -> dict[int, list[float]]:
    paths = sorted((LOG_DIR / group.log_group).glob(group.pattern), key=run_index)
    if len(paths) != 5:
        names = "\n".join(str(path.relative_to(ROOT)) for path in paths)
        raise RuntimeError(
            f"Expected 5 logs for {group.log_group}, found {len(paths)}:\n{names}"
        )
    return {run_index(path): parse_evaluations(path) for path in paths}


def align(curves: list[list[float]]) -> np.ndarray:
    length = min(len(curve) for curve in curves)
    return np.asarray([curve[:length] for curve in curves], dtype=float)


def save_all(fig: plt.Figure, stem: str) -> list[Path]:
    FIG_DIR.mkdir(exist_ok=True)
    paths: list[Path] = []
    for suffix in ("png", "svg", "pdf"):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_repeat_learning_curves(data: dict[str, dict[int, list[float]]]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9.6, 5.4))

    for group in GROUPS:
        curves = data[group.key]
        aligned = align(list(curves.values()))
        steps = np.arange(aligned.shape[1]) * EVAL_FREQ / 1_000_000
        mean = aligned.mean(axis=0)
        std = aligned.std(axis=0)

        for run, curve in curves.items():
            ax.plot(
                steps,
                curve[: aligned.shape[1]],
                color=group.color,
                linestyle=group.linestyle,
                alpha=0.16,
                linewidth=1.0,
            )
        ax.plot(
            steps,
            mean,
            color=group.color,
            linestyle=group.linestyle,
            linewidth=2.6,
            label=f"{group.label}, n={aligned.shape[0]}",
        )
        ax.fill_between(
            steps,
            mean - std,
            mean + std,
            color=group.color,
            alpha=0.10,
            linewidth=0,
        )

    ax.set_title("Hopper-v4 随机种子 10991：默认失败组与更新间隔调整组")
    ax.set_xlabel("环境交互步数（百万）")
    ax.set_ylabel("评估回报")
    ax.legend(loc="upper left")
    ax.margins(x=0.01)
    fig.tight_layout()
    return save_all(fig, "plif_hopper_seed10991_repeat_learning_curves")


def print_summary(data: dict[str, dict[int, list[float]]]) -> None:
    for group in GROUPS:
        curves = data[group.key]
        aligned = align(list(curves.values()))
        duplicated = np.allclose(aligned, aligned[0])
        print(group.label)
        print(f"  runs: {sorted(curves)}")
        print(f"  eval points: {aligned.shape[1]}")
        print(f"  max mean curve value: {aligned.mean(axis=0).max():.3f}")
        print(f"  final mean curve value: {aligned.mean(axis=0)[-1]:.3f}")
        print(f"  repeated runs overlap exactly: {duplicated}")


def main() -> None:
    configure_style()
    data = {group.key: read_group(group) for group in GROUPS}
    paths = plot_repeat_learning_curves(data)
    print_summary(data)
    print("\nGenerated figures:")
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
