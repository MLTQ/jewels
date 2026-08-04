"""Held-out sampled-render evaluation for structured jewel autoencoders."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.autoencoder import StructuredJewelAutoencoder
from sol.corpus import FeatureNormalizer, FittedExample
from sol.render import render_exact


@dataclass(frozen=True)
class ExampleMetric:
    name: str
    source_id: str
    psnr: float
    target_jewels: int
    decoded_jewels: int


@dataclass(frozen=True)
class EvaluationReport:
    mean_psnr: float
    median_psnr: float
    macro_source_psnr: float
    source_mean_psnr: dict[str, float]
    mean_count_ratio: float
    examples: tuple[ExampleMetric, ...]

    def to_dict(self) -> dict:
        return {
            "mean_psnr": self.mean_psnr,
            "median_psnr": self.median_psnr,
            "macro_source_psnr": self.macro_source_psnr,
            "source_mean_psnr": self.source_mean_psnr,
            "mean_count_ratio": self.mean_count_ratio,
            "examples": [metric.__dict__ for metric in self.examples],
        }


def select_balanced_examples(
    examples: list[FittedExample], max_examples: int
) -> list[FittedExample]:
    """Round-robin windows across source videos for deterministic validation."""
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    by_source: dict[str, list[FittedExample]] = {}
    for example in examples:
        by_source.setdefault(example.source_id, []).append(example)
    selected = []
    offset = 0
    source_ids = sorted(by_source)
    while len(selected) < max_examples:
        added = False
        for source_id in source_ids:
            group = by_source[source_id]
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) == max_examples:
                    break
        if not added:
            break
        offset += 1
    return selected


@torch.no_grad()
def evaluate_roundtrip(
    model: StructuredJewelAutoencoder,
    examples: list[FittedExample],
    normalizer: FeatureNormalizer,
    *,
    device: torch.device | str,
    points_per_example: int = 512,
    max_examples: int = 4,
    seed: int = 0,
) -> EvaluationReport:
    """Measure fit-render versus decoded-render PSNR on held-out source videos."""
    if points_per_example <= 0 or max_examples <= 0:
        raise ValueError("evaluation counts must be positive")
    if not examples:
        raise ValueError("evaluation requires held-out examples")
    target_device = torch.device(device)
    was_training = model.training
    model.eval()
    metrics = []
    selected = select_balanced_examples(examples, max_examples)
    for index, example in enumerate(selected):
        normalized = normalizer.normalize(example.features)[None].to(target_device)
        latents = model.encoder(normalized)
        decoded_normalized = model.decode(latents)[0]
        decoded = normalizer.denormalize(decoded_normalized)
        generator = torch.Generator(device=target_device).manual_seed(seed + index)
        points = torch.rand(
            points_per_example,
            3,
            generator=generator,
            device=target_device,
        ) * 2 - 1
        target_render = render_exact(example.features.to(target_device), points).clamp(0, 1)
        decoded_render = render_exact(decoded, points).clamp(0, 1)
        mse = (target_render - decoded_render).square().mean().clamp_min(1e-10)
        psnr = float(-10 * torch.log10(mse))
        metrics.append(
            ExampleMetric(
                name=example.name,
                source_id=example.source_id,
                psnr=psnr,
                target_jewels=example.features.shape[0],
                decoded_jewels=decoded.shape[0],
            )
        )
    if was_training:
        model.train()
    psnrs = sorted(metric.psnr for metric in metrics)
    middle = len(psnrs) // 2
    median = (
        psnrs[middle]
        if len(psnrs) % 2
        else 0.5 * (psnrs[middle - 1] + psnrs[middle])
    )
    ratios = [metric.decoded_jewels / metric.target_jewels for metric in metrics]
    source_values: dict[str, list[float]] = {}
    for metric in metrics:
        source_values.setdefault(metric.source_id, []).append(metric.psnr)
    source_means = {
        source_id: sum(values) / len(values)
        for source_id, values in sorted(source_values.items())
    }
    return EvaluationReport(
        mean_psnr=sum(psnrs) / len(psnrs),
        median_psnr=median,
        macro_source_psnr=sum(source_means.values()) / len(source_means),
        source_mean_psnr=source_means,
        mean_count_ratio=sum(ratios) / len(ratios),
        examples=tuple(metrics),
    )
