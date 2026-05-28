from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
LOG_DIR = ROOT / "logs"
FIG_DIR = ROOT / "figures"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

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


@dataclass(frozen=True)
class Algorithm:
    key: str
    label: str
    log_prefix: str
    color: str


ALGORITHMS = [
    Algorithm("ANN", "ANN", "TD3_ANN", "#4d4d4d"),
    Algorithm("LIF", "LIF", "PT_LIF_RERUN", "#2f6db3"),
    Algorithm("PLIF", "PLIF", "PT_PLIF_STABLE", "#c44e52"),
    Algorithm("CLIF", "CLIF", "PT_CLIF", "#55a868"),
]

MANUAL_MAX_EVAL = {
    # The TD3_ANN Hopper seed10991 log was deleted; this max-eval result was
    # manually restored from the recorded experiment result.
    ("ANN", "Hopper-v4", 10991): 3329.0,
}

EVAL_RE = re.compile(r"Evaluation over 10 episodes:\s*([-+]?\d+(?:\.\d+)?)")


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


def find_log(algorithm: Algorithm, env: str, seed: int) -> Path:
    pattern = f"{algorithm.log_prefix}*_{env}_seed{seed}_gpu*.log"
    matches = sorted(
        path
        for path in LOG_DIR.rglob(pattern)
        if "archive_unmatched_plif" not in path.parts
    )
    if not matches:
        raise FileNotFoundError(
            f"No log found for {algorithm.log_prefix} {env} seed {seed}"
        )
    if len(matches) > 1:
        names = "\n".join(str(path.relative_to(ROOT)) for path in matches)
        raise RuntimeError(
            f"Multiple logs found for {algorithm.log_prefix} {env} seed {seed}:\n{names}"
        )
    return matches[0]


def parse_evaluations(path: Path) -> list[float]:
    evaluations: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = EVAL_RE.search(line)
            if match:
                evaluations.append(float(match.group(1)))
    if not evaluations:
        raise ValueError(f"No evaluation records found in {path}")
    return evaluations


def read_curves() -> tuple[dict[str, dict[str, dict[int, list[float]]]], list[str]]:
    curves: dict[str, dict[str, dict[int, list[float]]]] = {
        algorithm.key: {env: {} for env in ENVS} for algorithm in ALGORITHMS
    }
    missing: list[str] = []
    for algorithm in ALGORITHMS:
        for env in ENVS:
            for seed in SEEDS:
                try:
                    path = find_log(algorithm, env, seed)
                except FileNotFoundError:
                    missing.append(f"{algorithm.label} {env} seed {seed}")
                    continue
                curves[algorithm.key][env][seed] = parse_evaluations(path)
    return curves, missing


def align_curves(curves: list[list[float]]) -> np.ndarray:
    length = min(len(curve) for curve in curves)
    return np.asarray([curve[:length] for curve in curves], dtype=float)


def max_eval_values(
    curves: dict[str, dict[str, dict[int, list[float]]]]
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    values: dict[str, dict[str, np.ndarray]] = {}
    manual: dict[str, dict[str, np.ndarray]] = {}
    for algorithm in ALGORITHMS:
        values[algorithm.key] = {}
        manual[algorithm.key] = {}
        for env in ENVS:
            env_values: list[float] = []
            env_manual: list[bool] = []
            for seed in SEEDS:
                if seed in curves[algorithm.key][env]:
                    env_values.append(max(curves[algorithm.key][env][seed]))
                    env_manual.append(False)
                    continue

                manual_value = MANUAL_MAX_EVAL.get((algorithm.key, env, seed))
                env_values.append(manual_value if manual_value is not None else np.nan)
                env_manual.append(manual_value is not None)

            values[algorithm.key][env] = np.asarray(env_values, dtype=float)
            manual[algorithm.key][env] = np.asarray(env_manual, dtype=bool)
    return values, manual


def save_all(fig: plt.Figure, stem: str) -> list[Path]:
    paths = []
    for suffix in ("png", "svg", "pdf"):
        path = FIG_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_max_eval(values: dict[str, dict[str, np.ndarray]]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12.4, 5.9))
    x = np.arange(len(ENVS), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(ALGORITHMS))

    for offset, algorithm in zip(offsets, ALGORITHMS):
        env_values = np.asarray([values[algorithm.key][env] for env in ENVS])
        means = np.nanmean(env_values, axis=1)
        stds = np.nanstd(env_values, axis=1)
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=stds,
            capsize=3,
            color=algorithm.color,
            alpha=0.78,
            label=algorithm.label,
            edgecolor="white",
            linewidth=0.8,
        )
        jitter = np.linspace(-width * 0.23, width * 0.23, len(SEEDS))
        for seed_index in range(len(SEEDS)):
            visible = ~np.isnan(env_values[:, seed_index])
            ax.scatter(
                x[visible] + offset + jitter[seed_index],
                env_values[visible, seed_index],
                s=21,
                color=algorithm.color,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

    ax.set_title("五个随机种子下的最大评估回报")
    ax.set_ylabel("最大评估回报")
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_LABELS[env] for env in ENVS], rotation=15, ha="right")
    ax.legend(ncol=4, loc="upper left")
    ax.margins(x=0.025)
    fig.tight_layout()
    return save_all(fig, "algorithm_max_eval")


def plot_learning_curves(
    curves: dict[str, dict[str, dict[int, list[float]]]]
) -> list[Path]:
    fig, axes = plt.subplots(3, 2, figsize=(12.6, 11.3), sharex=False)
    axes_flat = axes.ravel()

    for ax, env in zip(axes_flat, ENVS):
        for algorithm in ALGORITHMS:
            seed_curves = [
                curves[algorithm.key][env][seed]
                for seed in SEEDS
                if seed in curves[algorithm.key][env]
            ]
            if not seed_curves:
                continue
            aligned = align_curves(seed_curves)
            steps = np.arange(aligned.shape[1]) * EVAL_FREQ / 1_000_000
            avg = aligned.mean(axis=0)
            spread = aligned.std(axis=0)
            linewidth = 2.2 if algorithm.key in {"LIF", "PLIF"} else 1.8
            alpha = 0.14 if algorithm.key in {"LIF", "PLIF"} else 0.09
            ax.plot(
                steps,
                avg,
                color=algorithm.color,
                linewidth=linewidth,
                label=algorithm.label,
            )
            ax.fill_between(
                steps,
                avg - spread,
                avg + spread,
                color=algorithm.color,
                alpha=alpha,
                linewidth=0,
            )
        ax.set_title(ENV_LABELS[env])
        ax.set_xlabel("环境交互步数（百万）")
        ax.set_ylabel("评估回报")

    axes_flat[-1].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(
        "不同算法在各环境中的平均学习曲线",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save_all(fig, "algorithm_learning_curves")


def write_index(paths: list[Path], missing: list[str]) -> Path:
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
                    f'<a href="{path.name}">PNG</a> ',
                    f'<a href="{path.stem}.svg">SVG</a> ',
                    f'<a href="{path.stem}.pdf">PDF</a>',
                    "</p>",
                    "</section>",
                ]
            )
        )

    algorithm_sources = ", ".join(
        f"{algorithm.label}: {algorithm.log_prefix}" for algorithm in ALGORITHMS
    )
    missing_note = (
        "<p>Missing logs: "
        + "; ".join(missing)
        + ".</p>"
        if missing
        else "<p>All configured logs were found.</p>"
    )
    page = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Experiment Result Figures</title>",
            "<style>",
            "body { font-family: Arial, Helvetica, sans-serif; margin: 24px; color: #222; }",
            "section { margin: 28px 0 44px; }",
            "img { width: 100%; max-width: 1500px; border: 1px solid #ddd; }",
            "a { color: #2f6db3; margin-right: 12px; }",
            "</style>",
            "</head>",
            "<body>",
            "<h1>Experiment Result Figures</h1>",
            f"<p>Sources: {algorithm_sources}. Seeds: {', '.join(map(str, SEEDS))}.</p>",
            missing_note,
            *cards,
            "</body>",
            "</html>",
        ]
    )
    index_path = FIG_DIR / "experiment_results_index.html"
    index_path.write_text(page + "\n", encoding="utf-8")
    return index_path


def print_summary(
    values: dict[str, dict[str, np.ndarray]],
    manual: dict[str, dict[str, np.ndarray]],
) -> None:
    print("Max eval reward mean +/- std:")
    for env in ENVS:
        print(f"  {env}:")
        for algorithm in ALGORITHMS:
            env_values = values[algorithm.key][env]
            env_manual = manual[algorithm.key][env]
            valid = env_values[~np.isnan(env_values)]
            manual_count = int(env_manual.sum())
            mean = valid.mean()
            std = valid.std()
            suffix = f", manual={manual_count}" if manual_count else ""
            print(
                f"    {algorithm.label}: {mean:.3f} +/- {std:.3f} "
                f"(n={valid.size}{suffix})"
            )


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    configure_style()

    curves, missing = read_curves()
    values, manual = max_eval_values(curves)

    paths: list[Path] = []
    paths.extend(plot_max_eval(values))
    paths.extend(plot_learning_curves(curves))
    index_path = write_index(paths, missing)

    print_summary(values, manual)
    if missing:
        print("\nMissing logs:")
        for item in missing:
            print(f"  {item}")
    print("\nGenerated figures:")
    for path in paths:
        print(path.relative_to(ROOT))
    print(index_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
