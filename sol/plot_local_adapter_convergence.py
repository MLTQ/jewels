"""Plot long-run convergence of frozen-base local appearance adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def _log_series(path: Path, offset: int = 0) -> dict[str, list[dict[str, float]]]:
    training = []
    validation = []
    for row in _records(path):
        step = offset + int(row["step"])
        if "appearance_lpips" in row:
            training.append({
                "step": step,
                "appearance_lpips": float(row["appearance_lpips"]),
                "render_loss": float(row["render_loss"]),
                "out_of_range_percent": 100.0
                * float(row["render_out_of_range_fraction"]),
            })
        if "evaluation" in row:
            validation.append({
                "step": step,
                "macro_psnr": float(row["evaluation"]["macro_psnr"]),
            })
    return {"training": training, "validation": validation}


def _metric(root: Path, report: str, key: str) -> dict[str, float]:
    saved = json.loads((root / report / "report.json").read_text())
    row = saved["perceptual_macro"][key]
    return {"lpips": float(row["lpips"]), "psnr": float(row["psnr"])}


def collect_convergence(screen_root: Path, convergence_root: Path) -> dict:
    """Collect loss, fixed-validation, and shared exact-audit curves."""
    raw_first = _log_series(
        convergence_root / "raw_r2_lpips005_seed0_12k" / "train_log.jsonl"
    )
    raw_continuation = _log_series(
        convergence_root / "raw_r2_lpips005_seed0_continue4k" / "train_log.jsonl",
        offset=12000,
    )
    derivative_seed0 = _log_series(
        convergence_root
        / "derivative32_r2_lpips005_seed0_12k"
        / "train_log.jsonl"
    )
    derivative_seed1 = _log_series(
        convergence_root
        / "derivative32_r2_lpips005_seed1_12k"
        / "train_log.jsonl"
    )
    source = _metric(screen_root, "audit_final_seed0_400", "irregular_seed0")
    exact = {
        "raw local": [
            {"step": 0, **source},
            {
                "step": 400,
                **_metric(
                    screen_root, "audit_final_seed0_400", "irregular_seed2"
                ),
            },
            {
                "step": 4000,
                **_metric(
                    convergence_root,
                    "audit_raw_curve_8000",
                    "irregular_seed1",
                ),
            },
            {
                "step": 8000,
                **_metric(
                    convergence_root,
                    "audit_raw_curve_8000",
                    "irregular_seed2",
                ),
            },
            {
                "step": 12000,
                **_metric(
                    convergence_root,
                    "audit_raw_plateau_16000",
                    "irregular_seed1",
                ),
            },
            {
                "step": 16000,
                **_metric(
                    convergence_root,
                    "audit_raw_plateau_16000",
                    "irregular_seed2",
                ),
            },
        ],
        "derivative x32 seed 0": [
            {
                "step": step,
                **_metric(
                    convergence_root,
                    "audit_derivative_progress_seed0",
                    f"irregular_seed{index}",
                ),
            }
            for index, step in enumerate(
                (0, 400, 800, 1600, 3200, 4000, 8000, 12000)
            )
        ],
        "derivative x32 seed 1": [
            {"step": 0, **source},
            *[
                {
                    "step": step,
                    **_metric(
                        convergence_root,
                        "audit_derivative_replication_curve",
                        f"irregular_seed{index}",
                    ),
                }
                for index, step in zip(
                    (4, 5, 6), (4000, 8000, 12000), strict=True
                )
            ],
        ],
    }
    return {
        "training": {
            "raw local": raw_first["training"] + raw_continuation["training"],
            "derivative x32 seed 0": derivative_seed0["training"],
            "derivative x32 seed 1": derivative_seed1["training"],
        },
        "validation": {
            "raw local": raw_first["validation"]
            + raw_continuation["validation"],
            "derivative x32 seed 0": derivative_seed0["validation"],
            "derivative x32 seed 1": derivative_seed1["validation"],
        },
        "exact": exact,
    }


def _moving_average(values: list[float], window: int = 5) -> list[float]:
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        subset = values[start : index + 1]
        smoothed.append(sum(subset) / len(subset))
    return smoothed


def plot_convergence(evidence: dict, out: Path) -> None:
    """Render training and exact convergence with registered quality gates."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    colors = {
        "raw local": "#5e81ac",
        "derivative x32 seed 0": "#bf616a",
        "derivative x32 seed 1": "#d08770",
    }

    for name, rows in evidence["training"].items():
        steps = [row["step"] for row in rows]
        values = [row["appearance_lpips"] for row in rows]
        axes[0, 0].plot(
            steps,
            _moving_average(values),
            color=colors[name],
            linewidth=2,
            label=f"{name} (5-point mean)",
        )
    axes[0, 0].set(
        title="Train perceptual loss keeps falling after the 400-step screen",
        xlabel="Optimizer updates",
        ylabel="Train LPIPS term (lower is better)",
    )
    axes[0, 0].legend()

    for name, rows in evidence["validation"].items():
        axes[0, 1].plot(
            [row["step"] for row in rows],
            [row["macro_psnr"] for row in rows],
            marker="o",
            color=colors[name],
            label=name,
        )
    axes[0, 1].axhline(20.0, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set(
        title="Fixed validation reconstruction stays above the guardrail",
        xlabel="Optimizer updates",
        ylabel="Sampled validation PSNR (dB)",
    )
    axes[0, 1].legend()

    for name, rows in evidence["exact"].items():
        axes[1, 0].plot(
            [row["step"] for row in rows],
            [row["lpips"] for row in rows],
            marker="o",
            linewidth=2.5,
            color=colors[name],
            label=name,
        )
    axes[1, 0].axhline(0.70, color="#a3be8c", linestyle="--", linewidth=2,
                       label="pitch gate: LPIPS < 0.70")
    axes[1, 0].set(
        title="Exact seven-frame audit: longer training crosses the gate",
        xlabel="Cumulative optimizer updates",
        ylabel="Exact LPIPS (lower is better)",
    )
    axes[1, 0].legend()

    for name, rows in evidence["exact"].items():
        axes[1, 1].plot(
            [row["step"] for row in rows],
            [row["psnr"] for row in rows],
            marker="o",
            linewidth=2.5,
            color=colors[name],
            label=name,
        )
    axes[1, 1].axhline(20.0, color="black", linestyle="--", linewidth=1.5,
                       label="quality floor: 20 dB")
    axes[1, 1].set(
        title="Derivative evidence improves PSNR and LPIPS together",
        xlabel="Cumulative optimizer updates",
        ylabel="Exact PSNR (dB; higher is better)",
    )
    axes[1, 1].legend()

    figure.suptitle(
        "Frozen irregular Jewelfield appearance convergence — two-seed replication",
        fontsize=16,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-root", required=True)
    parser.add_argument("--convergence-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    evidence = collect_convergence(
        Path(args.screen_root), Path(args.convergence_root)
    )
    out = Path(args.out)
    plot_convergence(evidence, out)
    out.with_suffix(".json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
