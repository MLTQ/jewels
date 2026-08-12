"""Fixed-path guide/context controls for the initial-compatible mark flow."""

from __future__ import annotations

import torch

from sol.birth_mark_flow import BirthMarkFlowModel, birth_mark_flow_objective
from sol.scaffold_mark_data import ScaffoldMarkCorpus, rasterize_scaffold_context


def _mean_controls(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("mark-flow control aggregation requires at least one row")
    names = tuple(rows[0])
    if any(tuple(row) != names for row in rows):
        raise ValueError("mark-flow control rows must have identical keys")
    means = {
        name: sum(row[name] for row in rows) / len(rows) for name in names
    }
    means.update(
        {
            "shuffled_minus_correct": means["shuffled_scaffold"] - means["correct"],
            "null_minus_correct": means["null_scaffold"] - means["correct"],
            "no_context_minus_correct": means["no_context"] - means["correct"],
        }
    )
    return means


@torch.no_grad()
def evaluate_scaffold_mark_flow(
    model: BirthMarkFlowModel,
    corpus: ScaffoldMarkCorpus,
    guide_rasters: dict[tuple[str, int], torch.Tensor],
    *,
    device: str | torch.device,
    seed: int = 0,
) -> dict:
    """Compare correct, cross-class, null, and no-context fixed flow paths."""
    validation = sorted(
        corpus.validation, key=lambda source: (source.field.class_id, source.field.source_id)
    )
    if len({source.field.class_id for source in validation}) < 2:
        raise ValueError("shuffled scaffold controls require two validation classes")
    alternate = {}
    for offset, source in enumerate(validation):
        candidate = validation[(offset + 1) % len(validation)]
        if candidate.field.class_id == source.field.class_id:
            raise ValueError("shuffled scaffold must come from a different class")
        alternate[source.field.source_id] = candidate.field.source_id

    target_device = torch.device(device)
    generator = torch.Generator(device=target_device).manual_seed(seed)
    model.eval()
    rows = []
    initial_rows = []
    continuation_rows = []
    per_source: dict[str, list[dict[str, float]]] = {}
    for source in validation:
        source_rows = []
        for view in source.views:
            if not len(view.births.values):
                continue
            key = (source.field.source_id, view.index)
            alternate_key = (alternate[source.field.source_id], view.index)
            if key not in guide_rasters or alternate_key not in guide_rasters:
                raise ValueError("guide raster map is incomplete for scaffold controls")
            context = rasterize_scaffold_context(
                view.context_features,
                corpus.context_standardizer,
                stride_frames=corpus.stride_frames,
                grid_spec=corpus.grid_spec,
            ).to(target_device)
            target = corpus.birth_standardizer.normalize(view.births.values).to(
                target_device
            )
            cells = view.births.cell_indices.to(target_device)
            ranks = view.births.slot_indices.to(target_device)
            text = corpus.prompt_embeddings[
                source.field.evaluation_prompt_indices[0]
            ].to(target_device)
            guide = guide_rasters[key].to(target_device)
            shuffled = guide_rasters[alternate_key].to(target_device)
            noise = torch.randn(target.shape, device=target_device, generator=generator)
            flow_time = torch.rand(1, device=target_device, generator=generator)

            def loss(context_value: torch.Tensor, guide_value: torch.Tensor) -> float:
                return float(
                    birth_mark_flow_objective(
                        model,
                        context_value,
                        target,
                        cells,
                        ranks,
                        text,
                        noise=noise,
                        flow_time=flow_time,
                        guide_raster=guide_value,
                    )
                )

            record = {
                "correct": loss(context, guide),
                "shuffled_scaffold": loss(context, shuffled),
                "null_scaffold": loss(context, torch.zeros_like(guide)),
                "no_context": loss(torch.zeros_like(context), guide),
            }
            rows.append(record)
            source_rows.append(record)
            (initial_rows if view.frontier == 0 else continuation_rows).append(record)
        if source_rows:
            per_source[source.field.source_id] = _mean_controls(source_rows)
    if not rows or not initial_rows or not continuation_rows:
        raise ValueError("evaluation requires birth-bearing initial and continuation views")
    return {
        "validation_views": len(rows),
        "aggregate": _mean_controls(rows),
        "initial": _mean_controls(initial_rows),
        "continuation": _mean_controls(continuation_rows),
        "per_source": per_source,
    }
