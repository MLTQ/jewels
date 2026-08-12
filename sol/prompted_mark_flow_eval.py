"""Fixed-path prompt controls for stochastic jewel birth-mark continuation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from sol.birth_mark_flow import BirthMarkFlowModel, birth_mark_flow_objective
from sol.streaming_corpus import PromptedContinuationCorpus
from sol.streaming_data import rasterize_context


@dataclass(frozen=True)
class MarkFlowControls:
    correct: float
    shuffled: float
    null: float


@dataclass(frozen=True)
class PromptedMarkFlowEvaluation:
    validation_views: int
    full_context: MarkFlowControls
    text_only: MarkFlowControls

    def to_dict(self) -> dict:
        return asdict(self)


@torch.no_grad()
def evaluate_prompted_mark_flow(
    model: BirthMarkFlowModel,
    corpus: PromptedContinuationCorpus,
    *,
    device: torch.device | str,
    seed: int = 0,
    guide_rasters: dict[tuple[str, int], torch.Tensor] | None = None,
    guide_tokens: dict[tuple[str, int], torch.Tensor] | None = None,
) -> PromptedMarkFlowEvaluation:
    """Compare correct, different-class, and null text on identical flow paths."""
    target_device = torch.device(device)
    validation = sorted(corpus.validation, key=lambda example: example.class_id)
    classes = [example.class_id for example in validation]
    if len(classes) < 2:
        raise ValueError("shuffled prompt controls require at least two classes")
    prompt_by_class = {
        example.class_id: corpus.prompt_embeddings[example.evaluation_prompt_indices[0]].to(
            target_device
        )
        for example in validation
    }
    shuffled_class = {
        class_id: classes[(index + 1) % len(classes)]
        for index, class_id in enumerate(classes)
    }
    totals = {
        family: {name: 0.0 for name in ("correct", "shuffled", "null")}
        for family in ("full_context", "text_only")
    }
    views = 0
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=target_device).manual_seed(seed)
    for example in validation:
        for view in example.dataset.views:
            context = rasterize_context(
                view.context_features,
                corpus.context_standardizer,
                prefix_frames=example.dataset.prefix_frames,
                stride_frames=example.dataset.stride_frames,
                grid_shape=example.dataset.grid_spec.shape,
            ).to(target_device)
            target = corpus.birth_standardizer.normalize(view.births.values).to(
                target_device
            )
            if not len(target):
                continue
            cells = view.births.cell_indices.to(target_device)
            slots = view.births.slot_indices.to(target_device)
            guide = None
            if model.guide_dim:
                if guide_rasters is None or (example.source_id, view.index) not in guide_rasters:
                    raise ValueError("guided flow evaluation requires every view raster")
                guide = guide_rasters[(example.source_id, view.index)].to(target_device)
            tokens = None
            if model.guide_token_dim:
                if guide_tokens is None or (example.source_id, view.index) not in guide_tokens:
                    raise ValueError("multiscale flow evaluation requires every view token set")
                tokens = guide_tokens[(example.source_id, view.index)].to(target_device)
            noise = torch.randn(
                target.shape,
                device=target_device,
                generator=generator,
            )
            flow_time = torch.rand(1, device=target_device, generator=generator)
            conditions = {
                "correct": prompt_by_class[example.class_id],
                "shuffled": prompt_by_class[shuffled_class[example.class_id]],
                "null": None,
            }
            for family, raster in (
                ("full_context", context),
                ("text_only", torch.zeros_like(context)),
            ):
                for name, text in conditions.items():
                    totals[family][name] += float(
                        birth_mark_flow_objective(
                            model,
                            raster,
                            target,
                            cells,
                            slots,
                            text,
                            noise=noise,
                            flow_time=flow_time,
                            guide_raster=guide,
                            guide_tokens=tokens,
                        )
                    )
            views += 1
    if was_training:
        model.train()
    if not views:
        raise ValueError("validation corpus contains no birth-bearing views")
    return PromptedMarkFlowEvaluation(
        validation_views=views,
        full_context=MarkFlowControls(
            **{name: total / views for name, total in totals["full_context"].items()}
        ),
        text_only=MarkFlowControls(
            **{name: total / views for name, total in totals["text_only"].items()}
        ),
    )
