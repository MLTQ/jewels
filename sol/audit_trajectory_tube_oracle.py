"""Audit two-donor foreground/background trajectory-tube Jewel composition."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image
import torch

from sol.audit_jewel_casting_language import (
    _audit_candidate,
    _render,
    _seed_for,
    load_field_records,
)
from sol.audit_scene_block_constellation_oracle import _lowest_fits, scene_key
from sol.block_token_language import BlockTokenCodebook, encode_block_tokens
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.jewel_casting_language import histogram_cosine
from sol.prompt_jewel_caster import (
    active_cell_histogram,
    active_tokens_to_features,
    encode_active_jewel_tokens,
)
from sol.render_jewel_casting_language import _panel, _row
from sol.render_streaming_continuation import frame_points
from sol.token_grid import GridSpec
from sol.train_factorized_prompt_jewel_caster import select_prompt_splits
from sol.trajectory_tube_realizer import fit_trajectory_tube_realizer


ARMS = ("tube composite", "coherent ceiling", "wrong-object tube", "pooled null")


def _macro(rows: list[dict]) -> dict:
    output = {}
    for arm in ARMS:
        output[arm] = {
            "target_histogram_cosine": sum(
                row["arms"][arm]["target_histogram_cosine"] for row in rows
            ) / len(rows),
            "voxel_psnr_diagnostic": sum(
                row["arms"][arm]["audit"]["voxel_psnr_to_continuous_source"]
                for row in rows
            ) / len(rows),
            "cell_center_lock_fraction": sum(
                row["arms"][arm]["audit"]["center_irregularity"][
                    "cell_center_lock_fraction"
                ] for row in rows
            ) / len(rows),
        }
    output["tube composite"].update({
        "foreground_fraction": sum(
            row["arms"]["tube composite"]["realization"]["foreground_fraction"]
            for row in rows
        ) / len(rows),
        "background_fraction": sum(
            row["arms"]["tube composite"]["realization"]["background_fraction"]
            for row in rows
        ) / len(rows),
        "adjustment_fraction": sum(
            row["arms"]["tube composite"]["realization"]["adjustment_fraction"]
            for row in rows
        ) / len(rows),
    })
    return output


@torch.no_grad()
def evaluate_source_disjoint(
    realizer,
    records,
    scenes: torch.Tensor,
    programs: torch.Tensor,
    *,
    physical_codebook,
    training_sources: list[str],
    null_token: int,
    generation_jewels: int,
    voxel_points: int,
    frames: int,
    height: int,
    width: int,
    seed: int,
    image_path: Path,
) -> tuple[list[dict], dict]:
    """Render the frozen source-disjoint two-donor causal suite."""
    device = realizer.device
    spec = GridSpec((8, 8, 4), slots_per_cell=1)
    frame_indices = torch.linspace(0, frames - 1, 3).round().long()
    render_points = frame_points(frames, frame_indices, height, width, device=device)
    rows, image_rows = [], []
    for owner, record in enumerate(records):
        target = record.features.to(device)
        encoded_target = encode_active_jewel_tokens(target, physical_codebook)
        target_histogram = active_cell_histogram(
            target[:, :3], encoded_target, spec=spec,
            vocabulary_size=physical_codebook.vocabulary_size,
        )
        generator = torch.Generator(device=device).manual_seed(seed + _seed_for(record))
        points = torch.rand(voxel_points, 3, generator=generator, device=device) * 2 - 1
        reference_render = _render(target, points, record.background)
        scene = int(scenes[owner])
        program = programs[owner]
        null_program = torch.full_like(program, null_token)
        arm_conditions = {
            "tube composite": ("tube", scene, program, None),
            "coherent ceiling": ("coherent", scene, program, None),
            "wrong-object tube": ("tube", scene, program, (scene + 1) % 3),
            "pooled null": (
                "coherent", realizer.coherent.null_scene_token, null_program, None
            ),
        }
        candidates = {"target": target}
        arms = {}
        for arm, (kind, arm_scene, arm_program, foreground_scene) in arm_conditions.items():
            arm_generator = torch.Generator(device=device).manual_seed(
                seed + 10000 + _seed_for(record)
            )
            if kind == "tube":
                centers, tokens, realization = realizer.sample(
                    arm_scene, arm_program, generation_jewels,
                    generator=arm_generator,
                    foreground_scene_token=foreground_scene,
                )
            else:
                centers, tokens, realization = realizer.coherent.sample(
                    arm_scene, arm_program, generation_jewels,
                    generator=arm_generator,
                )
            features = active_tokens_to_features(centers, tokens, physical_codebook)
            candidates[arm] = features
            histogram = active_cell_histogram(
                centers, tokens, spec=spec,
                vocabulary_size=physical_codebook.vocabulary_size,
            )
            named_realization = dict(realization)
            for key in ("foreground_training_field", "background_training_field"):
                if key in named_realization:
                    named_realization[key + "_source_id"] = training_sources[
                        int(named_realization[key])
                    ]
            if "selected_training_field" in named_realization:
                named_realization["selected_training_field_source_id"] = training_sources[
                    int(named_realization["selected_training_field"])
                ]
            arms[arm] = {
                "target_histogram_cosine": histogram_cosine(
                    target_histogram, histogram
                ),
                "realization": named_realization,
                "audit": _audit_candidate(
                    target, features, reference_render=reference_render,
                    points=points, background=record.background, spec=spec,
                ),
            }
        rows.append({
            "source_id": record.source_id,
            "fit_seed": record.fit_seed,
            "scene_token": scene,
            "arms": arms,
        })
        label = scene_key(record)[0]
        for arm in ("target",) + ARMS:
            rendered = _render(
                candidates[arm], render_points, record.background
            ).reshape(3, height, width, 3)
            image_rows.append(_row([
                _panel(frame, f"{label} / {arm} / t={int(index)}")
                for frame, index in zip(rendered, frame_indices)
            ]))
        print("rendered trajectory tube", record.source_id, flush=True)
    sheet = Image.new(
        "RGB",
        (image_rows[0].width,
         sum(row.height for row in image_rows) + 3 * (len(image_rows) - 1)),
        "white",
    )
    offset = 0
    for row in image_rows:
        sheet.paste(row, (0, offset))
        offset += row.height + 3
    image_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(image_path)
    return rows, _macro(rows)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-checkpoint", required=True)
    parser.add_argument("--split-report", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-jewels", type=int, default=72000)
    parser.add_argument("--voxel-points", type=int, default=4096)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--width", type=int, default=108)
    args = parser.parse_args()
    if min(
        args.generation_jewels, args.voxel_points, args.frames,
        args.height, args.width,
    ) <= 0:
        raise ValueError("trajectory-tube audit settings must be positive")
    device = torch.device(args.device)
    seed = 20260912
    checkpoint = torch.load(args.block_checkpoint, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if (
        int(architecture["block_vocabulary_size"]) != 1024
        or tuple(architecture["block_shape"]) != (16, 16, 8)
    ):
        raise ValueError("Gate 2a9 requires the frozen fine K=1024 language")
    block_codebook = BlockTokenCodebook.from_state_dict(
        checkpoint["block_codebook"], device
    )
    training_programs = checkpoint["training_programs"].to(device)
    split_protocol = json.loads(Path(args.split_report).read_text())["protocol"]
    records = load_field_records([Path(root) for root in split_protocol["roots"]])
    training, validation = select_prompt_splits(
        records,
        set(split_protocol["validation_sources"]),
        set(split_protocol["training_sources"]),
    )
    training = sorted(training, key=lambda record: record.source_id)
    validation = sorted(validation, key=lambda record: (record.source_id, record.fit_seed))
    if [record.source_id for record in training] != list(checkpoint["training_sources"]):
        raise ValueError("fine programs do not align with registered training sources")
    keys = sorted({scene_key(record) for record in training})
    key_to_scene = {key: index for index, key in enumerate(keys)}
    training_scenes = torch.tensor(
        [key_to_scene[scene_key(record)] for record in training],
        dtype=torch.long, device=device,
    )
    physical_codebook = load_factorized_codebook(args.codebook, device)
    realizer, fit_report = fit_trajectory_tube_realizer(
        [record.features.to(device) for record in training],
        training_scenes,
        block_codebook=block_codebook,
        physical_codebook=physical_codebook,
        jitter_std=0.005,
        tube_radius=0.78,
    )
    null_token = realizer.coherent.most_frequent_nonempty_token(training_programs)
    heldout_records = _lowest_fits(validation)
    validation_programs = {
        record.path: encode_block_tokens(record.features.to(device), block_codebook)[0]
        for record in validation
    }
    programs = torch.stack([validation_programs[record.path] for record in heldout_records])
    scenes = torch.tensor(
        [key_to_scene[scene_key(record)] for record in heldout_records],
        dtype=torch.long, device=device,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    rows, macro = evaluate_source_disjoint(
        realizer,
        heldout_records,
        scenes,
        programs,
        physical_codebook=physical_codebook,
        training_sources=[record.source_id for record in training],
        null_token=null_token,
        generation_jewels=args.generation_jewels,
        voxel_points=args.voxel_points,
        frames=args.frames,
        height=args.height,
        width=args.width,
        seed=seed,
        image_path=output / "qualitative_source_disjoint.png",
    )
    composites = [row["arms"]["tube composite"]["realization"] for row in rows]
    checks = {
        "distinct_foreground_background_donors": all(
            row["foreground_training_field"] != row["background_training_field"]
            for row in composites
        ),
        "both_donors_contribute_at_least_20pct": all(
            min(row["foreground_fraction"], row["background_fraction"]) >= 0.20
            for row in composites
        ),
        "count_adjustment_below_15pct": max(
            row["adjustment_fraction"] for row in composites
        ) < 0.15,
        "all_counts_exact": all(
            row["arms"][arm]["realization"]["emitted_jewels"]
            == args.generation_jewels
            for row in rows for arm in ARMS
        ),
        "correct_histogram_beats_wrong_object_and_null": all(
            macro["tube composite"]["target_histogram_cosine"]
            > macro[arm]["target_histogram_cosine"]
            for arm in ("wrong-object tube", "pooled null")
        ),
        "generated_centers_not_grid_locked": max(
            row["arms"][arm]["audit"]["center_irregularity"][
                "cell_center_lock_fraction"
            ] for row in rows for arm in ARMS
        ) < 0.01,
        "all_generated_renders_finite": all(
            math.isfinite(
                row["arms"][arm]["audit"]["voxel_psnr_to_continuous_source"]
            ) for row in rows for arm in ARMS
        ),
    }
    report = {
        "schema": "trajectory-tube-compositional-oracle-v1",
        "protocol": {
            "block_checkpoint": args.block_checkpoint,
            "split_report": args.split_report,
            "scene_keys": keys,
            "training_sources": [record.source_id for record in training],
            "validation_sources": [record.source_id for record in heldout_records],
            "target_program_selects_training_donors": True,
            "target_contributes_jewel_rows": False,
            "two_distinct_training_programs_per_composite": True,
            "oracle_is_not_prompt_inference": True,
            "oracle_is_template_backed": True,
            "emitted_centroids_are_continuous": True,
            "generation_jewels": args.generation_jewels,
        },
        "fit": fit_report,
        "macro": macro,
        "source_disjoint_records": rows,
        "gate": {
            "checks": checks,
            "numeric_passed": all(checks.values()),
            "qualitative_recognizability_and_wrong_object_review_required": True,
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"macro": macro, "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    main()
