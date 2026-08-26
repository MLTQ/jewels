"""Learn an autoregressive text-to-scene/trajectory/background Jewel program."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LearnedTrajectoryProgram:
    scene_token: int
    foreground_token: int
    background_token: int


class LearnedTrajectorySpeaker(nn.Module):
    """Small conditional autoregressive program speaker."""

    def __init__(
        self,
        text_dimension: int,
        hidden_dimension: int,
        scene_tokens: int,
        source_tokens: int,
    ) -> None:
        super().__init__()
        if min(text_dimension, hidden_dimension, scene_tokens, source_tokens) <= 0:
            raise ValueError("speaker dimensions must be positive")
        self.text_dimension = text_dimension
        self.hidden_dimension = hidden_dimension
        self.scene_tokens = scene_tokens
        self.source_tokens = source_tokens
        self.text_projection = nn.Sequential(
            nn.Linear(text_dimension, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
        )
        self.scene_embedding = nn.Embedding(scene_tokens, hidden_dimension)
        self.source_embedding = nn.Embedding(source_tokens, hidden_dimension)
        self.scene_head = nn.Linear(hidden_dimension, scene_tokens)
        self.foreground_head = nn.Sequential(
            nn.Linear(hidden_dimension * 2, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, source_tokens),
        )
        self.background_head = nn.Sequential(
            nn.Linear(hidden_dimension * 3, hidden_dimension),
            nn.SiLU(),
            nn.Linear(hidden_dimension, source_tokens),
        )

    def forward(
        self,
        text: torch.Tensor,
        scene_tokens: torch.Tensor,
        foreground_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if text.ndim != 2 or text.shape[1] != self.text_dimension:
            raise ValueError("text embeddings have the wrong shape")
        if scene_tokens.shape != text.shape[:1] or foreground_tokens.shape != text.shape[:1]:
            raise ValueError("teacher-forced program rows must align with text")
        hidden = self.text_projection(text)
        scene = self.scene_embedding(scene_tokens)
        foreground = self.source_embedding(foreground_tokens)
        return {
            "scene_logits": self.scene_head(hidden),
            "foreground_logits": self.foreground_head(torch.cat([hidden, scene], dim=1)),
            "background_logits": self.background_head(
                torch.cat([hidden, scene, foreground], dim=1)
            ),
        }

    @staticmethod
    def _sample_logits(
        logits: torch.Tensor,
        *,
        generator: torch.Generator,
        temperature: float,
        top_k: int,
    ) -> int:
        values, indices = torch.topk(logits / temperature, min(top_k, logits.numel()))
        probability = values.softmax(dim=0)
        selected = int(torch.multinomial(probability, 1, generator=generator))
        return int(indices[selected])

    @torch.no_grad()
    def sample(
        self,
        text: torch.Tensor,
        *,
        generator: torch.Generator,
        temperature: float = 0.8,
        top_k: int = 6,
    ) -> LearnedTrajectoryProgram:
        if text.shape != (1, self.text_dimension):
            raise ValueError("sampling requires one text embedding")
        if temperature <= 0 or top_k <= 0:
            raise ValueError("sampling controls must be positive")
        hidden = self.text_projection(text)
        scene = int(self.scene_head(hidden)[0].argmax())
        scene_tensor = torch.tensor([scene], device=text.device)
        foreground_logits = self.foreground_head(torch.cat([
            hidden, self.scene_embedding(scene_tensor)
        ], dim=1))[0]
        foreground = self._sample_logits(
            foreground_logits,
            generator=generator,
            temperature=temperature,
            top_k=top_k,
        )
        foreground_tensor = torch.tensor([foreground], device=text.device)
        background_logits = self.background_head(torch.cat([
            hidden,
            self.scene_embedding(scene_tensor),
            self.source_embedding(foreground_tensor),
        ], dim=1))[0]
        background_logits[foreground] = -torch.inf
        background = self._sample_logits(
            background_logits,
            generator=generator,
            temperature=temperature,
            top_k=top_k,
        )
        return LearnedTrajectoryProgram(scene, foreground, background)


def trajectory_program_loss(
    predictions: dict[str, torch.Tensor],
    scene_tokens: torch.Tensor,
    foreground_tokens: torch.Tensor,
    background_tokens: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return summed autoregressive cross-entropy and detached components."""
    scene = F.cross_entropy(predictions["scene_logits"], scene_tokens)
    foreground = F.cross_entropy(predictions["foreground_logits"], foreground_tokens)
    background = F.cross_entropy(predictions["background_logits"], background_tokens)
    total = scene + foreground + background
    return total, {
        "total": float(total.detach()),
        "scene": float(scene.detach()),
        "foreground": float(foreground.detach()),
        "background": float(background.detach()),
    }
