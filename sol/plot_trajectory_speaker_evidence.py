"""Plot the causal and prompt-control evidence for native trajectory-token speech."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_evidence(root: Path) -> dict:
    """Load the frozen reports into a compact plotting payload."""
    def report(relative: str) -> dict:
        return json.loads((root / relative).read_text())

    addressed = report("addressed_scene_block_oracle_v1/report.json")
    prompt = report("prompt_trajectory_speaker_v1/report.json")
    learned_train = report("learned_trajectory_speaker_v1/report.json")
    learned_audit = report("learned_trajectory_speaker_v1/audit/report.json")
    progress = report("learned_trajectory_speaker_v1/progress.json")
    return {
        "addressed_token_improvement_percent": (
            addressed["improvement_over_global_posterior"]["token_nll_fraction"] * 100
        ),
        "recognizable_prompts_visual_review": {
            "global scene": 0,
            "addressed blocks": 0,
            "coherent source": 3,
            "two-donor tube": 3,
            "prompt-only program": 3,
            "learned program": 3,
        },
        "prompt_lookup": prompt["summary"],
        "learned": learned_audit["summary"],
        "learned_training": learned_train,
        "progress": progress,
    }


def plot_evidence(evidence: dict, destination: Path) -> None:
    """Write the four-panel quantitative evidence figure."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    labels = list(evidence["recognizable_prompts_visual_review"])
    values = list(evidence["recognizable_prompts_visual_review"].values())
    colors = ["#8b8b8b" if value == 0 else "#2f7d68" for value in values]
    axes[0, 0].barh(labels, values, color=colors)
    axes[0, 0].set_xlim(0, 3.15)
    axes[0, 0].set_xticks([0, 1, 2, 3])
    axes[0, 0].set_xlabel("Recognizable prompt classes / 3 (visual review)")
    axes[0, 0].set_title("Coherence, not local vocabulary, unlocks subjects")
    for index, value in enumerate(values):
        axes[0, 0].text(value + 0.05, index, str(value), va="center")

    lookup = evidence["prompt_lookup"]
    learned = evidence["learned"]
    groups = ["Exact prompt compiler", "Learned unseen paraphrase"]
    shuffled = [
        lookup["correct_minus_shuffled_generation"],
        learned["correct_minus_shuffled_generation"],
    ]
    null = [
        lookup["correct_minus_null_generation"],
        learned["correct_minus_null_generation"],
    ]
    x = [0, 1]
    width = 0.34
    axes[0, 1].bar([value - width / 2 for value in x], shuffled, width,
                   label="Correct − shuffled", color="#315f9e")
    axes[0, 1].bar([value + width / 2 for value in x], null, width,
                   label="Correct − null", color="#b46a32")
    axes[0, 1].axhline(0.02, color="#315f9e", linestyle="--", linewidth=1,
                       label="Shuffled gate 0.02")
    axes[0, 1].axhline(0.01, color="#b46a32", linestyle=":", linewidth=1,
                       label="Null gate 0.01")
    axes[0, 1].set_xticks(x, groups)
    axes[0, 1].set_ylabel("OpenCLIP cosine margin")
    axes[0, 1].set_title("Rendered prompt causality passes in both speakers")
    axes[0, 1].legend(fontsize=8)
    for positions, series in (([v - width / 2 for v in x], shuffled),
                              ([v + width / 2 for v in x], null)):
        for position, value in zip(positions, series):
            axes[0, 1].text(position, value + 0.002, f"{value:.3f}",
                            ha="center", va="bottom", fontsize=9)

    retrieval = [
        lookup["correct_top1"],
        learned["correct_top1"],
        learned["correct_scene_consistent_programs"],
    ]
    retrieval_labels = ["Exact prompt\nCLIP top-1", "Unseen paraphrase\nCLIP top-1",
                        "Learned program\nscene consistency"]
    axes[1, 0].bar(retrieval_labels, retrieval,
                   color=["#2f7d68", "#a75248", "#2f7d68"])
    axes[1, 0].axhline(6, color="#555555", linestyle="--", linewidth=1,
                       label="Frozen retrieval gate 6/9")
    axes[1, 0].set_ylim(0, 9.6)
    axes[1, 0].set_ylabel("Programs / 9")
    axes[1, 0].set_title("One learned-generalization criterion remains")
    axes[1, 0].legend(fontsize=8)
    for index, value in enumerate(retrieval):
        axes[1, 0].text(index, value + 0.2, f"{value}/9", ha="center")

    steps = [row["step"] for row in evidence["progress"]]
    for condition, color in (
        ("correct", "#2f7d68"), ("shuffled", "#a75248"), ("null", "#8b6cae")
    ):
        values = [
            row["conditions"][condition]["token_nll_macro"]
            for row in evidence["progress"]
        ]
        axes[1, 1].plot(steps, values, marker="o", markersize=3,
                        label=condition, color=color)
    best = evidence["learned_training"]["best_step"]
    stop = evidence["learned_training"]["completed_step"]
    axes[1, 1].axvline(best, color="#2f7d68", linestyle="--", linewidth=1)
    axes[1, 1].axvline(stop, color="#555555", linestyle=":", linewidth=1)
    axes[1, 1].annotate(f"best {best}", (best, 2.25), xytext=(best + 70, 2.45),
                        arrowprops={"arrowstyle": "->", "linewidth": 0.8})
    axes[1, 1].annotate(f"plateau stop {stop}", (stop, 2.85),
                        xytext=(stop - 340, 3.15),
                        arrowprops={"arrowstyle": "->", "linewidth": 0.8})
    axes[1, 1].set_xlabel("Training updates")
    axes[1, 1].set_ylabel("Held-out token NLL (lower is better)")
    axes[1, 1].set_title("Longer training confirms overfit, not undertraining")
    axes[1, 1].legend(fontsize=8)

    figure.suptitle(
        "Native Jewel trajectory language: proof and remaining gap",
        fontsize=15,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _middle_panel(sheet: Image.Image, row: int) -> Image.Image:
    panel_width = sheet.width // 3
    row_height = 168
    row_stride = 171
    return sheet.crop((panel_width, row * row_stride,
                       panel_width * 2, row * row_stride + row_height))


def build_proof_sheet(
    prompt_sheet: Path,
    learned_sheet: Path,
    destination: Path,
) -> None:
    """Pair correct/shuffled middle frames for all classes and both speakers."""
    prompt_image = Image.open(prompt_sheet).convert("RGB")
    learned_image = Image.open(learned_sheet).convert("RGB")
    panels = []
    for experiment, sheet in (
        ("Exact prompt compiler", prompt_image),
        ("Learned unseen paraphrase", learned_image),
    ):
        for scene, label in enumerate(("Ballerina", "Dog", "Welder")):
            panels.append((experiment, label,
                           _middle_panel(sheet, scene * 3),
                           _middle_panel(sheet, scene * 3 + 1)))
    panel_width, panel_height = panels[0][2].size
    header = 46
    left_label = 170
    gap = 8
    width = left_label + panel_width * 2 + gap
    height = header + len(panels) * (panel_height + gap)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((left_label + 8, 12), "Correct text", fill="black", font=font)
    draw.text((left_label + panel_width + gap + 8, 12),
              "Cyclic-shuffled text", fill="black", font=font)
    y = header
    previous_experiment = None
    for experiment, label, correct, shuffled in panels:
        if experiment != previous_experiment:
            draw.text((8, y + 8), experiment, fill="black", font=font)
            previous_experiment = experiment
        draw.text((8, y + 28), label, fill="black", font=font)
        canvas.paste(correct, (left_label, y))
        canvas.paste(shuffled, (left_label + panel_width + gap, y))
        y += panel_height + gap
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.out)
    evidence = load_evidence(root)
    plot_evidence(evidence, output / "trajectory_speaker_evidence.png")
    (output / "trajectory_speaker_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    build_proof_sheet(
        root / "prompt_trajectory_speaker_v1/qualitative_seed20260914.png",
        root / "learned_trajectory_speaker_v1/audit/qualitative_seed20260920.png",
        output / "trajectory_speaker_proof_sheet.png",
    )


if __name__ == "__main__":
    main()
