"""Low-capacity additive style/action language model for Jewel casts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sol.prompt_jewel_caster import ACTIVE_FACTORS
from sol.token_grid import GridSpec


@dataclass(frozen=True)
class AdditiveLanguageCounts:
    """Source-disjoint sufficient statistics for global/style/action factors."""

    global_cells: torch.Tensor
    style_cells: torch.Tensor
    action_cells: torch.Tensor
    global_tokens: torch.Tensor
    style_tokens: torch.Tensor
    action_tokens: torch.Tensor


class AdditivePromptJewelCaster:
    """Compose shrunken style and action count posteriors by product of experts."""

    def __init__(
        self,
        counts: AdditiveLanguageCounts,
        *,
        spec: GridSpec,
        token_concentration: float = 64.0,
        cell_concentration: float = 256.0,
    ) -> None:
        if token_concentration <= 0 or cell_concentration <= 0:
            raise ValueError("Dirichlet concentrations must be positive")
        self.counts = counts
        self.spec = spec
        self.token_concentration = token_concentration
        self.cell_concentration = cell_concentration
        if counts.global_tokens.ndim != 3:
            raise ValueError("global token counts must have shape (cell,role,vocabulary)")

    @property
    def vocabulary_size(self) -> int:
        return int(self.counts.global_tokens.shape[-1])

    def _global_cell_probability(self) -> torch.Tensor:
        counts = self.counts.global_cells.float()
        return (counts + 1.0) / (counts.sum() + len(counts))

    def _factor_cell_probability(
        self, counts: torch.Tensor, global_probability: torch.Tensor
    ) -> torch.Tensor:
        counts = counts.float()
        return (
            counts + self.cell_concentration * global_probability
        ) / (counts.sum() + self.cell_concentration)

    def _global_token_probability(self) -> torch.Tensor:
        counts = self.counts.global_tokens.float()
        return (counts + 0.5) / (
            counts.sum(dim=2, keepdim=True) + 0.5 * self.vocabulary_size
        )

    def _factor_token_probability(
        self, counts: torch.Tensor, global_probability: torch.Tensor
    ) -> torch.Tensor:
        counts = counts.float()
        return (
            counts + self.token_concentration * global_probability
        ) / (counts.sum(dim=2, keepdim=True) + self.token_concentration)

    def probabilities(
        self,
        style_index: int | None,
        action_index: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cell and cell/role token probabilities for one text-resolved prompt."""
        global_cells = self._global_cell_probability()
        global_tokens = self._global_token_probability()
        if style_index is None or action_index is None:
            return global_cells, global_tokens
        style_cells = self._factor_cell_probability(
            self.counts.style_cells[style_index], global_cells
        )
        action_cells = self._factor_cell_probability(
            self.counts.action_cells[action_index], global_cells
        )
        cell_logits = (
            style_cells.clamp_min(1e-12).log()
            + action_cells.clamp_min(1e-12).log()
            - global_cells.clamp_min(1e-12).log()
        )
        cell_probability = cell_logits.softmax(dim=0)
        style_tokens = self._factor_token_probability(
            self.counts.style_tokens[style_index], global_tokens
        )
        action_tokens = self._factor_token_probability(
            self.counts.action_tokens[action_index], global_tokens
        )
        token_logits = (
            style_tokens.clamp_min(1e-12).log()
            + action_tokens.clamp_min(1e-12).log()
            - global_tokens.clamp_min(1e-12).log()
        )
        return cell_probability, token_logits.softmax(dim=2)

    def negative_log_likelihood(
        self,
        centers: torch.Tensor,
        tokens: torch.Tensor,
        *,
        style_index: int | None,
        action_index: int | None,
    ) -> dict:
        cells = self.spec.cell_index(centers)
        cell_probability, token_probability = self.probabilities(
            style_index, action_index
        )
        cell_nll = -cell_probability[cells].clamp_min(1e-12).log().mean()
        token_nll = {}
        for role, name in enumerate(ACTIVE_FACTORS):
            probability = token_probability[cells, role, tokens[:, role]]
            token_nll[name] = float(-probability.clamp_min(1e-12).log().mean())
        return {
            "cell_nll": float(cell_nll),
            "token_nll": token_nll,
            "token_nll_macro": sum(token_nll.values()) / len(token_nll),
        }

    def sample(
        self,
        count: int,
        *,
        style_index: int | None,
        action_index: int | None,
        generator: torch.Generator,
        chunk: int = 8192,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample irregular continuous points and active marks from an additive prompt."""
        if count <= 0:
            raise ValueError("additive caster sample count must be positive")
        cell_probability, token_probability = self.probabilities(
            style_index, action_index
        )
        cells = torch.multinomial(
            cell_probability, count, replacement=True, generator=generator
        )
        gu, gv, gt = self.spec.shape
        t = cells % gt
        v = (cells // gt) % gv
        u = cells // (gv * gt)
        coordinates = torch.stack([u, v, t], dim=1).to(cell_probability)
        cell_size = cell_probability.new_tensor([2 / gu, 2 / gv, 2 / gt])
        jitter = torch.rand(
            count, 3, generator=generator, device=cell_probability.device
        )
        centers = -1.0 + coordinates * cell_size + jitter * cell_size
        outputs = []
        for role in range(len(ACTIVE_FACTORS)):
            role_tokens = []
            for start in range(0, count, chunk):
                selected_cells = cells[start : start + chunk]
                role_tokens.append(
                    torch.multinomial(
                        token_probability[selected_cells, role],
                        1,
                        generator=generator,
                    ).squeeze(1)
                )
            outputs.append(torch.cat(role_tokens))
        return centers, torch.stack(outputs, dim=1)


def accumulate_language_counts(
    samples: list[tuple[torch.Tensor, torch.Tensor, int, int]],
    *,
    spec: GridSpec,
    vocabulary_size: int,
    style_count: int,
    action_count: int,
) -> AdditiveLanguageCounts:
    """Accumulate global and factor-owned cell/token counts from sampled training fields."""
    device = samples[0][0].device
    roles = len(ACTIVE_FACTORS)
    global_cells = torch.zeros(spec.n_cells, device=device)
    style_cells = torch.zeros(style_count, spec.n_cells, device=device)
    action_cells = torch.zeros(action_count, spec.n_cells, device=device)
    global_tokens = torch.zeros(
        spec.n_cells, roles, vocabulary_size, device=device
    )
    style_tokens = torch.zeros(
        style_count, spec.n_cells, roles, vocabulary_size, device=device
    )
    action_tokens = torch.zeros(
        action_count, spec.n_cells, roles, vocabulary_size, device=device
    )
    for centers, tokens, style_index, action_index in samples:
        cells = spec.cell_index(centers)
        cell_counts = torch.bincount(cells, minlength=spec.n_cells).float()
        global_cells += cell_counts
        style_cells[style_index] += cell_counts
        action_cells[action_index] += cell_counts
        for role in range(roles):
            flat = cells * vocabulary_size + tokens[:, role]
            histogram = torch.bincount(
                flat, minlength=spec.n_cells * vocabulary_size
            ).reshape(spec.n_cells, vocabulary_size).float()
            global_tokens[:, role] += histogram
            style_tokens[style_index, :, role] += histogram
            action_tokens[action_index, :, role] += histogram
    return AdditiveLanguageCounts(
        global_cells, style_cells, action_cells,
        global_tokens, style_tokens, action_tokens,
    )
