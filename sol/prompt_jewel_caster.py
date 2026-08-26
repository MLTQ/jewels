"""Text-conditioned marked point process for native Jewel casting."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from sol.factorized_jewel_casting_language import FactorizedCodebook
from sol.hierarchical_jewel_decoder import _assign_individual_factor, _factor
from sol.token_grid import GridSpec


ACTIVE_FACTORS = ("covariance", "surface", "gradient")


def encode_active_jewel_tokens(
    features: torch.Tensor, codebook: FactorizedCodebook
) -> torch.Tensor:
    """Assign the three nonconstant token IDs to individual continuous Jewels."""
    if codebook.bundle_size != 1:
        raise ValueError("active Jewel tokens require a bundle-1 codebook")
    values = features.new_zeros(len(features), 22)
    mean = codebook.normalizer.intrinsic_mean.to(features)
    std = codebook.normalizer.intrinsic_std.to(features)
    values[:, 3:] = (features[:, 3:] - mean) / std
    tokens = []
    for name in ACTIVE_FACTORS:
        token, _ = _assign_individual_factor(
            values, _factor(codebook, name), codebook.count_weight
        )
        tokens.append(token)
    return torch.stack(tokens, dim=1)


def active_tokens_to_features(
    centers: torch.Tensor,
    tokens: torch.Tensor,
    codebook: FactorizedCodebook,
) -> torch.Tensor:
    """Decode prompt-emitted centroids and active token IDs into renderable Jewels."""
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("generated centers must have shape (N,3)")
    if tokens.shape != (len(centers), len(ACTIVE_FACTORS)):
        raise ValueError("generated active tokens have an incompatible shape")
    values = centers.new_zeros(len(centers), 22)
    for column, name in enumerate(ACTIVE_FACTORS):
        factor = _factor(codebook, name)
        values[:, list(factor.dimensions)] = factor.prototypes[:, 0].to(values)[
            tokens[:, column]
        ]
    features = centers.new_empty(len(centers), 22)
    features[:, :3] = centers
    mean = codebook.normalizer.intrinsic_mean.to(centers)
    std = codebook.normalizer.intrinsic_std.to(centers)
    features[:, 3:] = values[:, 3:] * std + mean
    return features


def active_cell_histogram(
    centers: torch.Tensor,
    tokens: torch.Tensor,
    *,
    spec: GridSpec,
    vocabulary_size: int,
) -> torch.Tensor:
    """Build a concatenated cell-conditional histogram for prompt retrieval."""
    cells = spec.cell_index(centers)
    histograms = []
    for column in range(tokens.shape[1]):
        flat = cells * vocabulary_size + tokens[:, column]
        histogram = torch.bincount(
            flat, minlength=spec.n_cells * vocabulary_size
        ).reshape(spec.n_cells, vocabulary_size).float()
        histogram = histogram / histogram.sum(dim=1, keepdim=True).clamp_min(1.0)
        histograms.append(histogram.flatten())
    return torch.cat(histograms)


class PromptCentroidGMM(nn.Module):
    """Predict a bounded continuous centroid mixture from frozen text features."""

    def __init__(self, text_dim: int = 384, hidden_dim: int = 256, components: int = 64) -> None:
        super().__init__()
        if min(text_dim, hidden_dim, components) <= 0:
            raise ValueError("centroid mixture dimensions must be positive")
        self.components = components
        self.network = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, components * 7),
        )

    def parameters_for(self, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw = self.network(text).reshape(-1, self.components, 7)
        mixture_logits = raw[:, :, 0]
        means = torch.tanh(raw[:, :, 1:4])
        log_std = -4.0 + 4.5 * torch.sigmoid(raw[:, :, 4:7])
        return mixture_logits, means, log_std

    def negative_log_likelihood(self, text: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        logits, means, log_std = self.parameters_for(text)
        difference = (centers[:, None] - means) / log_std.exp()
        component = -0.5 * (
            difference.square() + 2 * log_std + math.log(2 * math.pi)
        ).sum(dim=2)
        return -torch.logsumexp(F.log_softmax(logits, dim=1) + component, dim=1)

    @torch.no_grad()
    def sample(
        self,
        text: torch.Tensor,
        count: int,
        *,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if text.shape[0] != 1 or count <= 0:
            raise ValueError("centroid sampling needs one prompt and a positive count")
        logits, means, log_std = self.parameters_for(text)
        component = torch.multinomial(
            logits.softmax(dim=1)[0], count, replacement=True, generator=generator
        )
        noise = torch.randn(
            count, 3, device=text.device, generator=generator
        )
        centers = means[0, component] + log_std[0, component].exp() * noise
        return centers.clamp(-0.999, 0.999)


class PromptJewelCaster(nn.Module):
    """Cast continuous centroids and three Jewel marks directly from frozen text."""

    def __init__(
        self,
        *,
        text_dim: int = 384,
        vocabulary_size: int = 1024,
        hidden_dim: int = 512,
        depth: int = 4,
        mixture_components: int = 64,
    ) -> None:
        super().__init__()
        if min(text_dim, vocabulary_size, hidden_dim, depth, mixture_components) <= 0:
            raise ValueError("prompt caster dimensions must be positive")
        self.text_dim = text_dim
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.mixture_components = mixture_components
        self.centroid_density = PromptCentroidGMM(
            text_dim=text_dim, hidden_dim=256, components=mixture_components
        )
        coordinate_dim = 3 * (1 + 2 * 4)
        layers: list[nn.Module] = [
            nn.Linear(text_dim + coordinate_dim, hidden_dim),
            nn.SiLU(),
        ]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, len(ACTIVE_FACTORS) * vocabulary_size))
        self.token_network = nn.Sequential(*layers)

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

    def token_logits(self, text: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        if text.shape[0] != len(centers):
            raise ValueError("prompt rows and centroids must align")
        logits = self.token_network(
            torch.cat([text, self.coordinate_features(centers)], dim=1)
        )
        return logits.reshape(len(centers), len(ACTIVE_FACTORS), self.vocabulary_size)

    def loss(
        self,
        text: torch.Tensor,
        centers: torch.Tensor,
        tokens: torch.Tensor,
        *,
        density_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = self.token_logits(text, centers)
        token_loss = F.cross_entropy(
            logits.flatten(0, 1), tokens.flatten()
        )
        density_loss = self.centroid_density.negative_log_likelihood(
            text, centers
        ).mean()
        total = token_loss + density_weight * density_loss
        return total, {"token_nll": token_loss, "centroid_nll": density_loss}

    @torch.no_grad()
    def sample_tokens(
        self,
        text: torch.Tensor,
        centers: torch.Tensor,
        *,
        generator: torch.Generator,
        temperature: float = 0.9,
        top_k: int = 64,
        chunk: int = 8192,
    ) -> torch.Tensor:
        if temperature <= 0 or not 0 < top_k <= self.vocabulary_size:
            raise ValueError("token sampling parameters are invalid")
        outputs = []
        for start in range(0, len(centers), chunk):
            part = centers[start : start + chunk]
            prompts = text.expand(len(part), -1)
            logits = self.token_logits(prompts, part) / temperature
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
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "mixture_components": self.mixture_components,
        }


class FactorizedPromptJewelCaster(nn.Module):
    """Compose separate style/action text with a continuous spatial intensity field."""

    def __init__(
        self,
        *,
        text_dim: int = 384,
        vocabulary_size: int = 1024,
        hidden_dim: int = 512,
        depth: int = 4,
    ) -> None:
        super().__init__()
        if min(text_dim, vocabulary_size, hidden_dim, depth) <= 0:
            raise ValueError("factorized prompt caster dimensions must be positive")
        self.text_dim = text_dim
        self.vocabulary_size = vocabulary_size
        self.hidden_dim = hidden_dim
        self.depth = depth
        coordinate_dim = 3 * (1 + 2 * 4)
        self.style_projection = nn.Linear(text_dim, hidden_dim)
        self.action_projection = nn.Linear(text_dim, hidden_dim)
        self.coordinate_projection = nn.Sequential(
            nn.Linear(coordinate_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
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

    def hidden(
        self,
        style: torch.Tensor,
        action: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        if len(style) != len(action) or len(style) != len(centers):
            raise ValueError("factorized prompt and coordinate rows must align")
        combined = (
            self.style_projection(style)
            + self.action_projection(action)
            + self.coordinate_projection(self.coordinate_features(centers))
        )
        return self.trunk(self.normalization(combined))

    def token_logits(
        self, style: torch.Tensor, action: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        logits = self.token_head(self.hidden(style, action, centers))
        return logits.reshape(
            len(centers), len(ACTIVE_FACTORS), self.vocabulary_size
        )

    def intensity_logits(
        self, style: torch.Tensor, action: torch.Tensor, centers: torch.Tensor
    ) -> torch.Tensor:
        return self.intensity_head(self.hidden(style, action, centers)).squeeze(1)

    def loss(
        self,
        style: torch.Tensor,
        action: torch.Tensor,
        centers: torch.Tensor,
        tokens: torch.Tensor,
        negative_centers: torch.Tensor,
        *,
        density_weight: float = 0.1,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        token_nll = F.cross_entropy(
            self.token_logits(style, action, centers).flatten(0, 1),
            tokens.flatten(),
        )
        positive = self.intensity_logits(style, action, centers)
        negative = self.intensity_logits(style, action, negative_centers)
        density_nce = 0.5 * (
            F.binary_cross_entropy_with_logits(positive, torch.ones_like(positive))
            + F.binary_cross_entropy_with_logits(negative, torch.zeros_like(negative))
        )
        return token_nll + density_weight * density_nce, {
            "token_nll": token_nll,
            "density_nce": density_nce,
        }

    @torch.no_grad()
    def sample_centers(
        self,
        style: torch.Tensor,
        action: torch.Tensor,
        count: int,
        *,
        generator: torch.Generator,
        proposal_multiplier: int = 4,
        chunk: int = 16384,
    ) -> torch.Tensor:
        if style.shape[0] != 1 or action.shape[0] != 1 or count <= 0:
            raise ValueError("center sampling needs one prompt and positive count")
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
                    part,
                )
            )
        weights = torch.cat(logits).softmax(dim=0)
        selected = torch.multinomial(
            weights, count, replacement=False, generator=generator
        )
        return proposals[selected]

    @torch.no_grad()
    def sample_tokens(
        self,
        style: torch.Tensor,
        action: torch.Tensor,
        centers: torch.Tensor,
        *,
        generator: torch.Generator,
        temperature: float = 0.9,
        top_k: int = 64,
        chunk: int = 8192,
    ) -> torch.Tensor:
        if temperature <= 0 or not 0 < top_k <= self.vocabulary_size:
            raise ValueError("factorized token sampling parameters are invalid")
        outputs = []
        for start in range(0, len(centers), chunk):
            part = centers[start : start + chunk]
            logits = self.token_logits(
                style.expand(len(part), -1),
                action.expand(len(part), -1),
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
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "density": "continuous_fourier_intensity_nce",
            "conditioning": "additive_style_action_text",
        }
