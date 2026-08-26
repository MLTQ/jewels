"""Plot controlled validation curves for the two neural prompt-to-Jewel speakers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ARMS = ("correct", "shuffled", "null")


def load_curve(path: Path, expected_schema: str, density_key: str) -> dict:
    """Load aligned validation histories from one neural speaker report."""
    report = json.loads(path.read_text())
    if report.get("schema") != expected_schema:
        raise ValueError(f"unexpected prompt-caster schema in {path}")
    history = report["history"]
    if not history:
        raise ValueError("prompt-caster history is empty")
    return {
        "steps": [int(row["step"]) for row in history],
        "best_step": int(report["best_step"]),
        "token": {
            arm: [float(row["controls"][arm]["token_nll_macro"]) for row in history]
            for arm in ARMS
        },
        "density": {
            arm: [float(row["controls"][arm][density_key]) for row in history]
            for arm in ARMS
        },
    }


def plot(joint: dict, factorized: dict, output: Path) -> None:
    """Render token and centroid/density curves for both architectures."""
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = {"correct": "tab:green", "shuffled": "tab:blue", "null": "tab:orange"}
    panels = (
        (axes[0, 0], joint, "token", "Joint-text token validation", "macro token NLL"),
        (axes[1, 0], joint, "density", "Joint-text centroid validation", "centroid NLL"),
        (axes[0, 1], factorized, "token", "Factorized-text token validation", "macro token NLL"),
        (axes[1, 1], factorized, "density", "Factorized-text density validation", "density NCE"),
    )
    for axis, curve, key, title, ylabel in panels:
        for arm in ARMS:
            axis.plot(curve["steps"], curve[key][arm], "o-", color=colors[arm], label=arm)
        axis.axvline(curve["best_step"], color="black", linestyle="--", linewidth=1, label="retained checkpoint")
        axis.set(title=title, xlabel="optimizer updates", ylabel=ylabel)
        axis.legend()
    figure.suptitle("Prompt-to-Jewel neural validation curves (lower is better)", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", required=True)
    parser.add_argument("--factorized", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    joint = load_curve(
        Path(args.joint), "prompt-native-jewel-caster-gate-v1", "centroid_nll"
    )
    factorized = load_curve(
        Path(args.factorized),
        "factorized-prompt-native-jewel-caster-gate-v1",
        "density_nce",
    )
    plot(joint, factorized, Path(args.out))
    print(json.dumps({"joint": joint, "factorized": factorized}, indent=2))


if __name__ == "__main__":
    main()
