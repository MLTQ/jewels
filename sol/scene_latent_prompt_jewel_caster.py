"""Prompt-conditioned Jewel caster with one shared stochastic scene state."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from sol.prompt_jewel_caster import ACTIVE_FACTORS


class SceneLatentPromptJewelCaster(nn.Module):
    """Emit a scene state once, then condition every continuous Jewel cast on it."""

    def __init__(
        self,
        *,
        text_dim: int = 384,
        vocabulary_size: int = 1024,
        scene_dim: int = 32,
        hidden_dim: int = 512,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if min(text_dim, vocabulary_size, scene_dim, hidden_dim, depth) <= 0:
            raise ValueError("scene-latent caster dimensions must be positive")
        self.text_dim = text_dim
        self.vocabulary_size = vocabulary_size
        self.scene_dim = scene_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        coordinate_dim = 3 * (1 + 2 * 4)
        self.style_projection = nn.Linear(text_dim, hidden_dim)
        self.action_projection = nn.Linear(text_dim, hidden_dim)
        self.scene_projection = nn.Sequential(
            nn.Linear(scene_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(coordinate_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.prior_network = nn.Sequential(
            nn.Linear(text_dim * 2, hidden_dim // 2), nn.SiLU(),
            nn.Linear(hidden_dim // 2, scene_dim * 2),
        )
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        self.trunk = nn.Sequential(*layers)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.token_head = nn.Linear(
            hidden_dim, len(ACTIVE_FACTORS) * vocabulary_size
        )
        self.intensity_head = nn.Linear(hidden_dim, 1)

    def coordinate_features(self, centers: torch.Tensor) -> torch.Tensor:
        output = [centers]
        for frequency in (1.0, 2.0, 4.0, 8.0):
            output.extend(
                [
                    torch.sin(torch.pi * frequency * centers),
                    torch.cos(torch.pi * frequency * centers),
                ]
            )
        return torch.cat(output, dim=1)

    def prior_parameters(
        self, style: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if style.shape != action.shape or style.shape[-1] != self.text_dim:
            raise ValueError("scene prior requires aligned style/action embeddings")
        mean, raw_log_std = self.prior_network(torch.cat([style, action], dim=1)).chunk(
            2, dim=1
        )
        log_std = -2.5 + 2.5 * torch.sigmoid(raw_log_std)
        return mean, log_std

    def hidden(
        self,
        style: torch.Tensor,
        action: torch.Tensor,
        scene: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        if not (len(style) == len(action) == len(scene) == len(centers)):
            raise ValueError("prompt, scene, and coordinate rows must align")
        combined = (
            self.style_projection(style)
            + self.action_projection(action)
            + self.scene_projection(scene)
            + self.coordinate_projection(self.coordinate_features(centers))
        )
        return self.trunk(self.normalization(combined))

    def token_logits(
        self,
        style: torch.Tensor,
        action: torch.Tensor,
        scene: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.token_head(self.hidden(style, action, scene, centers))
        return logits.reshape(
            len(centers), len(ACTIVE_FACTORS), self.vocabulary_size
        )

    def intensity_logits(
        self,
        style: torch.Tensor,
        action: torch.Tensor,
        scene: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        return self.intensity_head(
            self.hidden(style, action, scene, centers)
        ).squeeze(1)

    def loss(
        self,
        style: torch.Tensor,
        action: torch.Tensor,
        scene: torch.Tensor,
        centers: torch.Tensor,
        tokens: torch.Tensor,
        negative_centers: torch.Tensor,
        *,
        density_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        token_nll = F.cross_entropy(
            self.token_logits(style, action, scene, centers).flatten(0, 1),
            tokens.flatten(),
        )
        positive = self.intensity_logits(style, action, scene, centers)
        negative = self.intensity_logits(
            style, action, scene, negative_centers
        )
        density_nce = 0.5 * (
            F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
            + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
        )
        return token_nll + density_weight * density_nce, {
            "token_nll": token_nll,
            "density_nce": density_nce,
        }

    @torch.no_grad()
    def sample_scene(
        self,
        style: torch.Tensor,
        action: torch.Tensor,
        *,
        generator: torch.Generator,
        epsilon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mean, log_std = self.prior_parameters(style, action)
        if epsilon is None:
            epsilon = torch.randn(
                mean.shape, device=mean.device, generator=generator
            )
        if epsilon.shape != mean.shape:
            raise ValueError("scene epsilon must match the prompt batch")
        return mean + log_std.exp() * epsilon

    @torch.no_grad()
    def sample_centers(
        self,
        style: torch.Tensor,
        action: torch.Tensor,
        scene: torch.Tensor,
        count: int,
        *,
        generator: torch.Generator,
        proposal_multiplier: int = 4,
        chunk: int = 16384,
    ) -> torch.Tensor:
        if style.shape[0] != 1 or action.shape[0] != 1 or scene.shape[0] != 1:
            raise ValueError("center sampling needs one prompt and one scene")
        if count <= 0:
            raise ValueError("center sampling count must be positive")
        proposal_count = count * proposal_multiplier
        proposals = torch.rand(
            proposal_count, 3, device=style.device, generator=generator
        ) * 1.998 - 0.999
        logits = []
        for start in range(0, proposal_count, chunk):
            part = proposals[start : start + chunk]
            logits.append(
                self.intensity_logits(
                    style.expand(len(part), -1),
                    action.expand(len(part), -1),
                    scene.expand(len(part), -1),
                    part,
                )
            )
        selected = torch.multinomial(
            torch.cat(logits).softmax(dim=0),
            count,
            replacement=False,
            generator=generator,
        )
        return proposals[selected]

    @torch.no_grad()
    def sample_tokens(
        self,
        style: torch.Tensor,
        action: torch.Tensor,
        scene: torch.Tensor,
        centers: torch.Tensor,
        *,
        generator: torch.Generator,
        temperature: float = 0.9,
        top_k: int = 64,
        chunk: int = 8192,
    ) -> torch.Tensor:
        if temperature <= 0 or not 0 < top_k <= self.vocabulary_size:
            raise ValueError("scene-latent token sampling parameters are invalid")
        outputs = []
        for start in range(0, len(centers), chunk):
            part = centers[start : start + chunk]
            logits = self.token_logits(
                style.expand(len(part), -1),
                action.expand(len(part), -1),
                scene.expand(len(part), -1),
                part,
            ) / temperature
            values, indices = torch.topk(logits, top_k, dim=2)
            sampled = torch.multinomial(
                values.softmax(dim=2).reshape(-1, top_k),
                1,
                generator=generator,
            ).reshape(len(part), len(ACTIVE_FACTORS))
            outputs.append(indices.gather(2, sampled[:, :, None]).squeeze(2))
        return torch.cat(outputs)

    def architecture(self) -> dict:
        return {
            "text_dim": self.text_dim,
            "vocabulary_size": self.vocabulary_size,
            "scene_dim": self.scene_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "density": "continuous_fourier_intensity_nce",
            "conditioning": "style_action_plus_shared_stochastic_scene",
        }
