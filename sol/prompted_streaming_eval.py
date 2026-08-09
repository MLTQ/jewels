"""Correct, shuffled, and null text controls for direct jewel continuation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sol.streaming_corpus import PromptedContinuationCorpus
from sol.streaming_data import BirthTarget, rasterize_context
from sol.streaming_model import BirthContinuationModel


@dataclass(frozen=True)
class PromptControlMetrics:
    feature_mse: float
    count_mae: float

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class PromptedStreamingEvaluation:
    full_context: dict[str, PromptControlMetrics]
    text_only: dict[str, PromptControlMetrics]
    validation_views: int

    def to_dict(self) -> dict:
        return {
            "full_context": {
                name: metric.to_dict() for name, metric in self.full_context.items()
            },
            "text_only": {
                name: metric.to_dict() for name, metric in self.text_only.items()
            },
            "validation_views": self.validation_views,
        }


def _device_target(view, dataset, device: torch.device) -> BirthTarget:
    births = view.births
    return BirthTarget(
        values=dataset.birth_standardizer.normalize(births.values).to(device),
        cell_indices=births.cell_indices.to(device),
        slot_indices=births.slot_indices.to(device),
        counts=births.counts.to(device),
        global_ids=births.global_ids.to(device),
        birth_frames=births.birth_frames.to(device),
    )


def _context(view, dataset, device: torch.device) -> torch.Tensor:
    return rasterize_context(
        view.context_features,
        dataset.context_standardizer,
        prefix_frames=dataset.prefix_frames,
        stride_frames=dataset.stride_frames,
        grid_shape=dataset.grid_spec.shape,
    ).to(device)


@torch.no_grad()
def evaluate_prompted_streaming(
    model: BirthContinuationModel,
    corpus: PromptedContinuationCorpus,
    *,
    device: str | torch.device,
) -> PromptedStreamingEvaluation:
    """Hold ranks fixed while swapping only text and optional prefix context."""
    target_device = torch.device(device)
    model.eval()
    validation = sorted(corpus.validation, key=lambda example: example.class_id)
    if len(validation) < 2:
        raise ValueError("shuffled text control requires at least two validation classes")
    prompt_by_class = {
        example.class_id: corpus.prompt_embeddings[example.evaluation_prompt_indices[0]].to(
            target_device
        )
        for example in validation
    }
    classes = sorted(prompt_by_class)
    shuffled_class = {
        class_id: classes[(index + 1) % len(classes)]
        for index, class_id in enumerate(classes)
    }
    errors = {
        context_mode: {
            condition: {"feature": [], "count": []}
            for condition in ("correct", "shuffled", "null")
        }
        for context_mode in ("full_context", "text_only")
    }
    views = 0
    for example in validation:
        correct = prompt_by_class[example.class_id]
        shuffled = prompt_by_class[shuffled_class[example.class_id]]
        for view in example.dataset.views:
            views += 1
            context = _context(view, example.dataset, target_device)
            target = _device_target(view, example.dataset, target_device)
            for context_mode, context_raster in (
                ("full_context", context),
                ("text_only", torch.zeros_like(context)),
            ):
                for condition, text in (
                    ("correct", correct),
                    ("shuffled", shuffled),
                    ("null", None),
                ):
                    output = model.forward_training(context_raster, target, text)
                    feature = (
                        float(F.mse_loss(output.occupied_features, target.values))
                        if len(target.values)
                        else 0.0
                    )
                    predicted_counts = output.log_count.clamp_min(0).expm1()
                    count = float(
                        F.l1_loss(predicted_counts, target.counts.float())
                    )
                    errors[context_mode][condition]["feature"].append(feature)
                    errors[context_mode][condition]["count"].append(count)

    def summarize(context_mode: str) -> dict[str, PromptControlMetrics]:
        result = {}
        for condition, values in errors[context_mode].items():
            result[condition] = PromptControlMetrics(
                feature_mse=sum(values["feature"]) / len(values["feature"]),
                count_mae=sum(values["count"]) / len(values["count"]),
            )
        return result

    return PromptedStreamingEvaluation(
        full_context=summarize("full_context"),
        text_only=summarize("text_only"),
        validation_views=views,
    )
