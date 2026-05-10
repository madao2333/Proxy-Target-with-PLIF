from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-chenjunwei")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
SEEDS = [10991, 22297, 33431, 75183, 95718]
EVAL_FREQ = 5000

ALG_COLORS = {
    "LIF": "#2f6db3",
    "PLIF": "#c44e52",
}
TAU_COLORS = {
    "hidden h0": "#2f6db3",
    "hidden h1": "#55a868",
    "output": "#c44e52",
}

EVAL_RE = re.compile(r"Evaluation over 10 episodes:\s*([-+]?\d+(?:\.\d+)?)")
HIDDEN_TAU_RE = re.compile(
    r"Current PLIF tau \(hidden\): h0: ([0-9.]+), h1: ([0-9.]+)"
)
OUTPUT_TAU_RE = re.compile(r"Current PLIF tau \(output\): ([0-9.]+)")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
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
        }
    )


def read_side_by_side() -> dict[str, dict[int, dict[str, float]]]:
    path = LOG_DIR / "PT_LIF_vs_PT_PLIF_max_eval_side_by_side.tsv"
    data: dict[str, dict[int, dict[str, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            env = row["experiment"]
            seed = int(row["seed"])
            data.setdefault(env, {})[seed] = {
                "LIF": float(row["LIF_max_value"]),
                "PLIF": float(row["PLIF_max_value"]),
            }
    return data


def find_log(algorithm: str, env: str, seed: int) -> Path:
    matches = sorted(LOG_DIR.glob(f"PT_{algorithm}_{env}_seed{seed}_gpu*.log"))
    if not matches:
        raise FileNotFoundError(f"No log found for {algorithm} {env} seed {seed}")
    return matches[0]


def parse_log(path: Path) -> tuple[list[float], list[tuple[float, float, float]]]:
    evaluations: list[float] = []
    taus: list[tuple[float, float, float]] = []
    pending_hidden: tuple[float, float] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            eval_match = EVAL_RE.search(line)
            if eval_match:
                evaluations.append(float(eval_match.group(1)))
                continue

            hidden_match = HIDDEN_TAU_RE.search(line)
            if hidden_match:
                pending_hidden = (
                    float(hidden_match.group(1)),
                    float(hidden_match.group(2)),
                )
                continue

            output_match = OUTPUT_TAU_RE.search(line)
            if output_match and pending_hidden is not None:
                taus.append((*pending_hidden, float(output_match.group(1))))
                pending_hidden = None

    return evaluations, taus


def read_curves() -> dict[str, dict[str, dict[int, list[float]]]]:
    curves: dict[str, dict[str, dict[int, list[float]]]] = {
        "LIF": {env: {} for env in ENVS},
        "PLIF": {env: {} for env in ENVS},
    }
    for algorithm in ("LIF", "PLIF"):
        for env in ENVS:
            for seed in SEEDS:
                evaluations, _ = parse_log(find_log(algorithm, env, seed))
                curves[algorithm][env][seed] = evaluations
    return curves


def read_plif_taus() -> dict[str, dict[int, list[tuple[float, float, float]]]]:
    tau_data: dict[str, dict[int, list[tuple[float, float, float]]]] = {
        env: {} for env in ENVS
    }
    for env in ENVS:
        for seed in SEEDS:
            _, taus = parse_log(find_log("PLIF", env, seed))
            tau_data[env][seed] = taus
    return tau_data


def align_curves(curves: list[list[float]]) -> np.ndarray:
    length = min(len(curve) for curve in curves if curve)
    return np.asarray([curve[:length] for curve in curves], dtype=float)


def align_tau_layer(
    tau_curves: list[list[tuple[float, float, float]]], layer_index: int
) -> np.ndarray:
    length = min(len(curve) for curve in tau_curves if curve)
    return np.asarray(
        [[curve[index][layer_index] for index in range(length)] for curve in tau_curves],
        dtype=float,
    )


def save_all(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for suffix in ("png", "svg", "pdf"):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_max_eval(data: dict[str, dict[int, dict[str, float]]]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    x = np.arange(len(ENVS), dtype=float)
    width = 0.32

    for offset, algorithm in [(-width / 2, "LIF"), (width / 2, "PLIF")]:
        values = np.asarray(
            [[data[env][seed][algorithm] for seed in SEEDS] for env in ENVS],
            dtype=float,
        )
        means = values.mean(axis=1)
        stds = values.std(axis=1)
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=ALG_COLORS[algorithm],
            alpha=0.78,
            label=algorithm,
            edgecolor="white",
            linewidth=0.8,
        )
        jitter = np.linspace(-width * 0.25, width * 0.25, len(SEEDS))
        for seed_index in range(len(SEEDS)):
            ax.scatter(
                x + offset + jitter[seed_index],
                values[:, seed_index],
                s=24,
                color=ALG_COLORS[algorithm],
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )

    ax.set_title("Max Evaluation: LIF vs PLIF")
    ax.set_ylabel("Max eval reward")
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS[env] for env in ENVS], rotation=15, ha="right")
    ax.legend(ncol=2, loc="upper left")
    ax.margins(x=0.03)
    fig.tight_layout()
    return save_all(fig, "lif_plif_max_eval")


def plot_learning_curves(curves: dict[str, dict[str, dict[int, list[float]]]]) -> list[Path]:
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 11.2), sharex=False)
    axes_flat = axes.ravel()

    for ax, env in zip(axes_flat, ENVS):
        for algorithm in ("LIF", "PLIF"):
            seed_curves = [curves[algorithm][env][seed] for seed in SEEDS]
            aligned = align_curves(seed_curves)
            steps = np.arange(aligned.shape[1]) * EVAL_FREQ / 1_000_000
            avg = aligned.mean(axis=0)
            spread = aligned.std(axis=0)
            ax.plot(steps, avg, color=ALG_COLORS[algorithm], linewidth=2.0, label=algorithm)
            ax.fill_between(
                steps,
                avg - spread,
                avg + spread,
                color=ALG_COLORS[algorithm],
                alpha=0.14,
                linewidth=0,
            )
        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Env steps (M)")
        ax.set_ylabel("Eval reward")

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle("Learning Curves: LIF vs PLIF", y=0.995, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save_all(fig, "lif_plif_learning_curves")


def plot_tau_curves(tau_data: dict[str, dict[int, list[tuple[float, float, float]]]]) -> list[Path]:
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 11.2), sharex=False)
    axes_flat = axes.ravel()
    layers = [("hidden h0", 0), ("hidden h1", 1), ("output", 2)]

    for ax, env in zip(axes_flat, ENVS):
        seed_taus = [tau_data[env][seed] for seed in SEEDS]
        for layer_name, layer_index in layers:
            aligned = align_tau_layer(seed_taus, layer_index)
            steps = np.arange(aligned.shape[1]) * EVAL_FREQ / 1_000_000
            avg = aligned.mean(axis=0)
            spread = aligned.std(axis=0)
            ax.plot(steps, avg, color=TAU_COLORS[layer_name], linewidth=2.0, label=layer_name)
            ax.fill_between(
                steps,
                avg - spread,
                avg + spread,
                color=TAU_COLORS[layer_name],
                alpha=0.10,
                linewidth=0,
            )
        ax.axhline(0.75, color="#666666", linestyle=":", linewidth=1.1, label="LIF fixed tau")
        ax.set_ylim(0.45, 1.02)
        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("Env steps (M)")
        ax.set_ylabel("Tau")

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle("PLIF Tau Curves", y=0.995, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save_all(fig, "plif_tau_curves")


def plot_walker2d_seed_curves(curves: dict[str, dict[str, dict[int, list[float]]]]) -> list[Path]:
    env = "Walker2d-v4"
    fig, axes = plt.subplots(len(SEEDS), 1, figsize=(11.2, 11.6), sharex=True)

    for ax, seed in zip(axes, SEEDS):
        for algorithm in ("LIF", "PLIF"):
            values = np.asarray(curves[algorithm][env][seed], dtype=float)
            steps = np.arange(values.size) * EVAL_FREQ / 1_000_000
            ax.plot(steps, values, color=ALG_COLORS[algorithm], linewidth=1.8, label=algorithm)
        ax.set_ylabel(f"seed {seed}\nreward")
        ax.grid(True, alpha=0.25, linestyle="--")

    axes[-1].set_xlabel("Env steps (M)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(
        "Walker2d Learning Curves by Seed",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save_all(fig, "walker2d_seed_learning_curves")


def write_index(paths: list[Path]) -> Path:
    png_paths = [path for path in paths if path.suffix == ".png"]
    cards = []
    for path in png_paths:
        title = path.stem.replace("_", " ").title()
        cards.append(
            "\n".join(
                [
                    "<section>",
                    f"<h2>{title}</h2>",
                    f'<img src="{path.name}" alt="{title}">',
                    "<p>",
                    f'<a href="{path.stem}.png">PNG</a> ',
                    f'<a href="{path.stem}.svg">SVG</a> ',
                    f'<a href="{path.stem}.pdf">PDF</a>',
                    "</p>",
                    "</section>",
                ]
            )
        )

    page = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>LIF vs PLIF Figures</title>",
            "<style>",
            "body { font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #222; }",
            "section { margin: 28px 0 44px; }",
            "img { width: 100%; max-width: 1400px; border: 1px solid #ddd; }",
            "a { color: #2f6db3; margin-right: 12px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>LIF vs PLIF Figures</h1>",
            "<p>Generated with matplotlib from logs and summary TSV files.</p>",
            *cards,
            "</body>",
            "</html>",
        ]
    )
    index_path = FIG_DIR / "index.html"
    index_path.write_text(page + "\n", encoding="utf-8")
    return index_path


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    configure_style()

    max_data = read_side_by_side()
    curves = read_curves()
    tau_data = read_plif_taus()

    paths: list[Path] = []
    paths.extend(plot_max_eval(max_data))
    paths.extend(plot_learning_curves(curves))
    paths.extend(plot_tau_curves(tau_data))
    paths.extend(plot_walker2d_seed_curves(curves))
    index_path = write_index(paths)

    print("Generated figures:")
    for path in paths:
        print(path.relative_to(ROOT))
    print(index_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
