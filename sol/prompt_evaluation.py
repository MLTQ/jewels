"""Evaluate held-out prompt-template separation before jewel model training."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sol.prompt_embeddings import PromptEmbeddingCache, load_prompt_cache, manifest_digest


@dataclass(frozen=True)
class ClassPromptMetric:
    class_name: str
    evaluation_prompt: str
    predicted_class: str
    correct_cosine: float
    best_wrong_cosine: float
    margin: float
    mean_train_paraphrase_cosine: float


@dataclass(frozen=True)
class PromptGeometryReport:
    accuracy: float
    mean_margin: float
    minimum_margin: float
    maximum_cross_class_centroid_cosine: float
    classes: tuple[ClassPromptMetric, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["classes"] = [asdict(item) for item in self.classes]
        return payload


def evaluate_prompt_geometry(
    manifest: dict, cache: PromptEmbeddingCache
) -> PromptGeometryReport:
    """Classify unseen templates against class centroids made from training templates."""
    if cache.manifest_sha256 != manifest_digest(manifest):
        raise ValueError("prompt cache does not match the supplied manifest")
    lookup = {prompt: index for index, prompt in enumerate(cache.prompts)}
    class_names = [item["class_name"] for item in manifest["classes"]]
    centroids = []
    train_indices_by_class = []
    evaluation_indices_by_class = []
    for item in manifest["classes"]:
        train_indices = [lookup[text] for text in item["train_prompts"]]
        evaluation_indices = [lookup[text] for text in item["evaluation_prompts"]]
        train_indices_by_class.append(train_indices)
        evaluation_indices_by_class.append(evaluation_indices)
        centroid = cache.embeddings[train_indices].mean(dim=0)
        centroids.append(F.normalize(centroid, dim=0))
    centroid_tensor = torch.stack(centroids)
    centroid_similarity = centroid_tensor @ centroid_tensor.T
    cross_centroid = centroid_similarity[
        ~torch.eye(len(class_names), dtype=torch.bool)
    ]

    metrics = []
    correct = 0
    for class_index, class_name in enumerate(class_names):
        train_values = cache.embeddings[train_indices_by_class[class_index]]
        train_similarity = train_values @ train_values.T
        train_off_diagonal = train_similarity[
            ~torch.eye(len(train_values), dtype=torch.bool)
        ]
        for evaluation_index in evaluation_indices_by_class[class_index]:
            similarities = cache.embeddings[evaluation_index] @ centroid_tensor.T
            predicted = int(similarities.argmax())
            correct += predicted == class_index
            wrong = torch.cat(
                (similarities[:class_index], similarities[class_index + 1 :])
            )
            correct_cosine = float(similarities[class_index])
            best_wrong = float(wrong.max())
            metrics.append(
                ClassPromptMetric(
                    class_name=class_name,
                    evaluation_prompt=cache.prompts[evaluation_index],
                    predicted_class=class_names[predicted],
                    correct_cosine=correct_cosine,
                    best_wrong_cosine=best_wrong,
                    margin=correct_cosine - best_wrong,
                    mean_train_paraphrase_cosine=float(train_off_diagonal.mean()),
                )
            )
    margins = [item.margin for item in metrics]
    return PromptGeometryReport(
        accuracy=correct / len(metrics),
        mean_margin=sum(margins) / len(margins),
        minimum_margin=min(margins),
        maximum_cross_class_centroid_cosine=float(cross_centroid.max()),
        classes=tuple(metrics),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    report = evaluate_prompt_geometry(manifest, load_prompt_cache(args.embeddings))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    print(json.dumps(report.to_dict(), indent=2), flush=True)


if __name__ == "__main__":
    main()
