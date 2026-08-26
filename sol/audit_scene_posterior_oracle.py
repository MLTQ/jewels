"""Diagnose whether the shared-scene prior or local Jewel decoder causes averaging."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F

from sol.audit_jewel_casting_language import _audit_candidate, _render, _seed_for, load_field_records
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.jewel_casting_language import histogram_cosine
from sol.prompt_jewel_caster import (
    ACTIVE_FACTORS,
    active_cell_histogram,
    active_tokens_to_features,
    encode_active_jewel_tokens,
)
from sol.render_jewel_casting_language import _panel, _row
from sol.render_streaming_continuation import frame_points
from sol.scene_latent_prompt_jewel_caster import SceneLatentPromptJewelCaster
from sol.token_grid import GridSpec
from sol.train_prompt_jewel_caster import _metadata, frozen_text_embeddings


ARMS = ("posterior oracle", "text prior", "prompt blind")


def select_oracle_sources(records: list, training_sources: list[str]) -> list:
    """Select the lexicographically first registered source for each exact prompt/style."""
    by_id = {record.source_id: record for record in records}
    missing = set(training_sources) - set(by_id)
    if missing:
        raise ValueError(f"missing registered oracle sources: {sorted(missing)}")
    groups: dict[tuple[str, str], list] = {}
    for source_id in training_sources:
        record = by_id[source_id]
        metadata = _metadata(record.path)
        groups.setdefault(
            (metadata["style"], metadata["source_prompt"]), []
        ).append(record)
    return [
        min(group, key=lambda record: record.source_id)
        for _, group in sorted(groups.items())
    ]


def _teacher_metrics(
    model: SceneLatentPromptJewelCaster,
    style: torch.Tensor,
    action: torch.Tensor,
    scene: torch.Tensor,
    centers: torch.Tensor,
    tokens: torch.Tensor,
    negative: torch.Tensor,
) -> dict:
    logits = model.token_logits(
        style.expand(len(centers), -1),
        action.expand(len(centers), -1),
        scene.expand(len(centers), -1), centers,
    )
    token_nll = {
        name: float(F.cross_entropy(logits[:, index], tokens[:, index]))
        for index, name in enumerate(ACTIVE_FACTORS)
    }
    positive = model.intensity_logits(
        style.expand(len(centers), -1),
        action.expand(len(centers), -1),
        scene.expand(len(centers), -1), centers,
    )
    negative_logits = model.intensity_logits(
        style.expand(len(negative), -1),
        action.expand(len(negative), -1),
        scene.expand(len(negative), -1), negative,
    )
    density = 0.5 * (
        F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
        + F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
    )
    return {
        "token_nll": token_nll,
        "token_nll_macro": sum(token_nll.values()) / len(token_nll),
        "density_nce": float(density),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--generation-jewels", type=int, default=72000)
    parser.add_argument("--height", type=int, default=72)
    parser.add_argument("--width", type=int, default=108)
    parser.add_argument("--frames", type=int, default=49)
    args = parser.parse_args()
    if min(args.generation_jewels, args.height, args.width, args.frames) <= 0:
        raise ValueError("oracle render settings must be positive")
    device = torch.device(args.device)
    seed = 20260904
    report = json.loads(Path(args.report).read_text())
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    architecture = saved["architecture"]
    model = SceneLatentPromptJewelCaster(
        text_dim=int(architecture["text_dim"]),
        vocabulary_size=int(architecture["vocabulary_size"]),
        scene_dim=int(architecture["scene_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        depth=int(architecture["depth"]),
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    posterior = saved["training_posterior_mean"].to(device)
    training_sources = list(report["protocol"]["training_sources"])
    if posterior.shape[0] != len(training_sources):
        raise ValueError("posterior rows do not align with report training sources")
    posterior_index = {source: index for index, source in enumerate(training_sources)}
    records = load_field_records([Path(root) for root in args.root])
    selected = select_oracle_sources(records, training_sources)
    styles = list(saved["styles"])
    actions = list(saved["actions"])
    embeddings = frozen_text_embeddings(
        [f"{style} visual style" for style in styles] + actions,
        device=device, model_name=saved["text_model"],
    )
    style_embedding = {
        value: embeddings[index : index + 1]
        for index, value in enumerate(styles)
    }
    action_embedding = {
        value: embeddings[len(styles) + index : len(styles) + index + 1]
        for index, value in enumerate(actions)
    }
    codebook = load_factorized_codebook(args.codebook, device)
    spec = GridSpec(codebook.grid_shape, slots_per_cell=1)
    indices = torch.linspace(0, args.frames - 1, 3).round().long()
    render_points = frame_points(
        args.frames, indices, args.height, args.width, device=device
    )
    rows = []
    qualitative_rows = []
    for record in selected:
        metadata = _metadata(record.path)
        style = style_embedding[metadata["style"]]
        action = action_embedding[metadata["source_prompt"]]
        null_style, null_action = torch.zeros_like(style), torch.zeros_like(action)
        prior_mean, _ = model.prior_parameters(style, action)
        null_mean, _ = model.prior_parameters(null_style, null_action)
        scenes = {
            "posterior oracle": posterior[posterior_index[record.source_id] : posterior_index[record.source_id] + 1],
            "text prior": prior_mean,
            "prompt blind": null_mean,
        }
        prompts = {
            "posterior oracle": (style, action),
            "text prior": (style, action),
            "prompt blind": (null_style, null_action),
        }
        target = record.features.to(device)
        target_tokens = encode_active_jewel_tokens(target, codebook)
        target_histogram = active_cell_histogram(
            target[:, :3], target_tokens, spec=spec,
            vocabulary_size=codebook.vocabulary_size,
        )
        generator = torch.Generator(device=device).manual_seed(seed + _seed_for(record))
        selected_rows = torch.randperm(
            len(target), generator=generator, device=device
        )[: min(16384, len(target))]
        target_sample = target[selected_rows]
        token_sample = target_tokens[selected_rows]
        negative = torch.rand(
            len(target_sample), 3, generator=generator, device=device
        ) * 1.998 - 0.999
        voxel_points = torch.rand(
            4096, 3, generator=generator, device=device
        ) * 2 - 1
        reference_render = _render(target, voxel_points, record.background)
        arms = {}
        generated_features = {}
        for arm in ARMS:
            arm_style, arm_action = prompts[arm]
            scene = scenes[arm]
            arm_generator = torch.Generator(device=device).manual_seed(
                seed + 10000 + _seed_for(record)
            )
            centers = model.sample_centers(
                arm_style, arm_action, scene, args.generation_jewels,
                generator=arm_generator, proposal_multiplier=4,
            )
            tokens = model.sample_tokens(
                arm_style, arm_action, scene, centers,
                generator=arm_generator, temperature=0.9, top_k=64,
            )
            features = active_tokens_to_features(centers, tokens, codebook)
            generated_features[arm] = features
            histogram = active_cell_histogram(
                centers, tokens, spec=spec,
                vocabulary_size=codebook.vocabulary_size,
            )
            arms[arm] = {
                "teacher_forced": _teacher_metrics(
                    model, arm_style, arm_action, scene,
                    target_sample[:, :3], token_sample, negative,
                ),
                "target_histogram_cosine": histogram_cosine(
                    target_histogram, histogram
                ),
                "audit": _audit_candidate(
                    target, features, reference_render=reference_render,
                    points=voxel_points, background=record.background, spec=spec,
                ),
            }
        rows.append({"source_id": record.source_id, "arms": arms})
        candidates = {"target": target, **generated_features}
        label = metadata["style"]
        for arm in ("target",) + ARMS:
            rendered = _render(
                candidates[arm], render_points, record.background
            ).reshape(3, args.height, args.width, 3)
            qualitative_rows.append(_row([
                _panel(frame, f"{label} / {arm} / t={int(index)}")
                for frame, index in zip(rendered, indices)
            ]))
        print("audited oracle", record.source_id, flush=True)

    def macro(arm: str, path: tuple[str, ...]) -> float:
        values = []
        for row in rows:
            value = row["arms"][arm]
            for key in path:
                value = value[key]
            values.append(float(value))
        return sum(values) / len(values)

    macro_report = {
        arm: {
            "token_nll_macro": macro(arm, ("teacher_forced", "token_nll_macro")),
            "density_nce": macro(arm, ("teacher_forced", "density_nce")),
            "target_histogram_cosine": macro(arm, ("target_histogram_cosine",)),
            "voxel_psnr_diagnostic": macro(
                arm, ("audit", "voxel_psnr_to_continuous_source")
            ),
        }
        for arm in ARMS
    }
    token_improvement = (
        macro_report["text prior"]["token_nll_macro"]
        - macro_report["posterior oracle"]["token_nll_macro"]
    ) / macro_report["text prior"]["token_nll_macro"]
    histogram_margin = (
        macro_report["posterior oracle"]["target_histogram_cosine"]
        - macro_report["text prior"]["target_histogram_cosine"]
    )
    checks = {
        "posterior_token_nll_improves_at_least_2pct": token_improvement >= 0.02,
        "posterior_histogram_margin_at_least_002": histogram_margin >= 0.02,
        "all_generated_renders_finite": all(
            math.isfinite(row["arms"][arm]["audit"]["voxel_psnr_to_continuous_source"])
            for row in rows for arm in ARMS
        ),
    }
    conclusion = (
        "text_scene_prior_bottleneck"
        if checks["posterior_token_nll_improves_at_least_2pct"]
        and checks["posterior_histogram_margin_at_least_002"]
        else "local_independent_decoder_still_bottlenecked"
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": "scene-posterior-oracle-audit-v1",
        "protocol": {
            "training_report": args.report,
            "selected_sources": [record.source_id for record in selected],
            "generation_jewels": args.generation_jewels,
            "oracle_uses_training_source_posterior": True,
            "oracle_is_not_valid_prompt_inference": True,
        },
        "macro": macro_report,
        "posterior_token_improvement_fraction": token_improvement,
        "posterior_histogram_margin": histogram_margin,
        "checks": checks,
        "causal_conclusion": conclusion,
        "records": rows,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    sheet = Image.new(
        "RGB",
        (qualitative_rows[0].width,
         sum(row.height for row in qualitative_rows) + 3 * (len(qualitative_rows) - 1)),
        "white",
    )
    offset = 0
    for row in qualitative_rows:
        sheet.paste(row, (0, offset))
        offset += row.height + 3
    sheet.save(output / "qualitative.png")
    print(json.dumps({
        "macro": macro_report, "checks": checks,
        "causal_conclusion": conclusion,
    }, indent=2))


if __name__ == "__main__":
    main()
