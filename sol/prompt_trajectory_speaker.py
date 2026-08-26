"""Compile registered prompts and seeds into finite trajectory-token Jewel programs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class PromptTrajectoryProgram:
    """One prompt-only scene, foreground, and background token utterance."""

    prompt: str
    scene_token: int
    foreground_token: int
    background_token: int
    seed: int
    condition: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PromptTrajectorySpeaker:
    """Deterministically compile text and randomness without target-field input."""

    prompts: tuple[str, ...]
    scene_sources: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if len(self.prompts) < 2 or len(self.prompts) != len(self.scene_sources):
            raise ValueError("speaker prompts and scene vocabularies must align")
        if len(set(self.prompts)) != len(self.prompts):
            raise ValueError("speaker prompts must be unique")
        flat = [source for rows in self.scene_sources for source in rows]
        if any(len(rows) < 2 for rows in self.scene_sources) or len(flat) != len(set(flat)):
            raise ValueError("each scene needs distinct source tokens")

    def scene_for_prompt(self, prompt: str) -> int:
        try:
            return self.prompts.index(" ".join(prompt.split()))
        except ValueError as error:
            raise ValueError("prompt is outside the registered Gate 2b0 vocabulary") from error

    def _program(
        self, prompt: str, scene: int, seed: int, condition: str
    ) -> PromptTrajectoryProgram:
        sources = self.scene_sources[scene]
        generator = torch.Generator().manual_seed(seed + 1009 * scene)
        order = torch.randperm(len(sources), generator=generator)
        foreground = sources[int(order[0])]
        background = sources[int(order[1])]
        return PromptTrajectoryProgram(
            prompt=prompt,
            scene_token=scene,
            foreground_token=foreground,
            background_token=background,
            seed=seed,
            condition=condition,
        )

    def compile(self, prompt: str, seed: int) -> PromptTrajectoryProgram:
        normalized = " ".join(prompt.split())
        return self._program(
            normalized, self.scene_for_prompt(normalized), seed, "correct prompt"
        )

    def compile_shuffled(self, prompt: str, seed: int) -> PromptTrajectoryProgram:
        scene = (self.scene_for_prompt(prompt) + 1) % len(self.prompts)
        return self._program(
            self.prompts[scene], scene, seed, "cyclic-shuffled prompt"
        )

    def compile_null(self, seed: int) -> PromptTrajectoryProgram:
        scene = seed % len(self.prompts)
        return self._program("", scene, seed, "null prompt")
