import argparse
from pathlib import Path

import numpy as np


def resolve_result_file(file_arg: str | None) -> Path:
    results_dir = Path("results")
    if file_arg:
        result_path = Path(file_arg)
        if not result_path.is_absolute():
            result_path = Path.cwd() / result_path
        return result_path

    npy_files = sorted(results_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError("results 目录下没有找到 .npy 文件")
    if len(npy_files) > 1:
        available = "\n".join(str(path) for path in npy_files)
        raise ValueError(f"发现多个结果文件，请用 --file 指定具体文件：\n{available}")
    return npy_files[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="查看训练生成的 .npy 评估结果")
    parser.add_argument("--file", default=None, help="要读取的 .npy 文件路径，不传时自动读取 results 目录下唯一的 .npy 文件")
    parser.add_argument("--plot", action="store_true", help="使用 matplotlib 绘制评估曲线")
    args = parser.parse_args()

    result_file = resolve_result_file(args.file)
    evaluations = np.load(result_file, allow_pickle=True)

    rewards: np.ndarray
    taus: np.ndarray | None = None

    if evaluations.ndim == 1:
        # Legacy format: [reward0, reward1, ...]
        if evaluations.dtype == object and len(evaluations) > 0:
            first_item = evaluations[0]
            if isinstance(first_item, (list, tuple, np.ndarray)) and len(first_item) >= 2:
                records = np.asarray(evaluations, dtype=np.float32)
                rewards = records[:, 0].astype(float)
                taus = records[:, 1].astype(float)
            else:
                rewards = evaluations.astype(float)
        else:
            rewards = evaluations.astype(float)
    elif evaluations.ndim == 2 and evaluations.shape[1] == 1:
        rewards = evaluations[:, 0].astype(float)
    elif evaluations.ndim == 2 and evaluations.shape[1] >= 2:
        # New format: [[reward, tau_0, tau_1, ...], ...]
        rewards = evaluations[:, 0].astype(float)
        taus = evaluations[:, 1:].astype(float)
    else:
        raise ValueError("不支持的结果格式，期望一维 reward 数组或二维 [reward, tau...] 数组")

    print(f"文件: {result_file}")
    print(f"评估次数: {len(rewards)}")
    print(f"初始值: {float(rewards[0]):.3f}")
    print(f"最新值: {float(rewards[-1]):.3f}")
    print(f"最大值: {float(np.max(rewards)):.3f}")
    print(f"最小值: {float(np.min(rewards)):.3f}")
    if taus is not None:
        if taus.ndim == 1:
            valid_taus = taus[~np.isnan(taus)]
            if valid_taus.size > 0:
                print(f"tau 初始值: {float(valid_taus[0]):.6f}")
                print(f"tau 最新值: {float(valid_taus[-1]):.6f}")
                print(f"tau 最大值: {float(np.max(valid_taus)):.6f}")
                print(f"tau 最小值: {float(np.min(valid_taus)):.6f}")
        else:
            layer_count = taus.shape[1]
            print(f"tau 层数: {layer_count}")
            print("各层 tau 统计:")
            for layer_index in range(layer_count):
                layer_taus = taus[:, layer_index]
                valid_layer_taus = layer_taus[~np.isnan(layer_taus)]
                if valid_layer_taus.size > 0:
                    print(
                        f"  layer {layer_index:02d}: "
                        f"initial={float(valid_layer_taus[0]):.6f}, "
                        f"latest={float(valid_layer_taus[-1]):.6f}, "
                        f"max={float(np.max(valid_layer_taus)):.6f}, "
                        f"min={float(np.min(valid_layer_taus)):.6f}"
                    )
    print("\n全部评估结果:")

    for index, reward in enumerate(rewards):
        if taus is None:
            print(f"{index:03d}: reward={float(reward):.3f}")
        elif taus.ndim == 1:
            if np.isnan(taus[index]):
                print(f"{index:03d}: reward={float(reward):.3f}")
            else:
                print(f"{index:03d}: reward={float(reward):.3f}, tau={float(taus[index]):.6f}")
        else:
            tau_values = ", ".join(f"{float(value):.6f}" for value in taus[index])
            print(f"{index:03d}: reward={float(reward):.3f}, taus=[{tau_values}]")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("未安装 matplotlib，无法使用 --plot") from exc

        plt.plot(rewards, marker="o", linewidth=1.5, label="reward")
        if taus is not None and np.any(~np.isnan(taus)):
            if taus.ndim == 1:
                plt.plot(taus, marker="s", linewidth=1.2, label="tau")
            else:
                for layer_index in range(taus.shape[1]):
                    plt.plot(taus[:, layer_index], marker="s", linewidth=1.2, label=f"tau_layer_{layer_index}")
        plt.title(result_file.stem)
        plt.xlabel("Evaluation Index")
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()