"""Load the frozen Jewel grammar and turn prompt programs into playable MP4s."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import torch
import torch.nn.functional as F

from sol.audit_jewel_casting_language import _featurizer, load_field_records
from sol.audit_scene_block_constellation_oracle import scene_key
from sol.block_token_language import BlockTokenCodebook
from sol.factorized_jewel_casting_language import load_factorized_codebook
from sol.learned_trajectory_speaker import LearnedTrajectorySpeaker
from sol.prompt_jewel_caster import active_tokens_to_features
from sol.prompt_trajectory_speaker import PromptTrajectorySpeaker
from sol.render_streaming_continuation import frame_points
from sol.semantic_trajectory_realizer import fit_semantic_trajectory_realizer
from sol.train_factorized_prompt_jewel_caster import select_prompt_splits


DEMO_VERSION = "prompt-video-demo-v1"


@dataclass(frozen=True)
class PromptVideoPaths:
    """Artifact paths required to reconstruct the frozen generation stack."""

    block_checkpoint: Path
    split_report: Path
    physical_codebook: Path
    learned_speaker: Path | None = None

    @classmethod
    def from_project_root(cls, root: Path) -> "PromptVideoPaths":
        base = root / "sol" / "results" / "jewel_casting_language_v0"
        return cls(
            block_checkpoint=base / "fine_block_language_v1" / "language.pt",
            split_report=base / "prompt_shared_scene_scaling_v1" / "r6" / "report.json",
            physical_codebook=(
                base / "hierarchical_v1" / "gate0d" / "codebook_individual_k1024.pt"
            ),
            learned_speaker=(
                base / "learned_trajectory_speaker_v1" / "speaker.pt"
            ),
        )


@dataclass(frozen=True)
class GeneratedJewelField:
    """One prompt-emitted field plus the provenance required to render it."""

    features: torch.Tensor
    background: torch.Tensor
    metadata: dict[str, Any]


def normalize_prompt(prompt: str) -> str:
    """Normalize whitespace and reject empty or unbounded demo inputs."""
    normalized = " ".join(prompt.split())
    if not normalized:
        raise ValueError("Please enter a prompt.")
    if len(normalized) > 300:
        raise ValueError("Prompts are limited to 300 characters in this demo.")
    return normalized


def video_basename(prompt: str, mode: str, seed: int) -> str:
    """Return a stable, filesystem-safe name for one deterministic request."""
    if mode not in {"exact", "learned"}:
        raise ValueError("mode must be 'exact' or 'learned'")
    digest = hashlib.sha256(
        f"{DEMO_VERSION}\0{mode}\0{seed}\0{normalize_prompt(prompt)}".encode()
    ).hexdigest()[:12]
    words = "-".join(
        "".join(character for character in word.lower() if character.isalnum())
        for word in normalize_prompt(prompt).split()[:6]
    ).strip("-")
    return f"{words or 'prompt'}-seed{seed}-{mode}-{digest}"


def realization_seed(mode: str, declared_seed: int, scene_token: int) -> int:
    """Reproduce the frozen exact/learned field-RNG ownership rules."""
    if mode not in {"exact", "learned"}:
        raise ValueError("mode must be 'exact' or 'learned'")
    condition_offset = 0 if mode == "exact" else 500000
    return declared_seed + condition_offset + 100000 * scene_token


def ffmpeg_command(
    binary: str, output: Path, *, width: int, height: int, fps: int
) -> list[str]:
    """Build the raw-RGB to browser-compatible H.264 encoder command."""
    if min(width, height, fps) <= 0:
        raise ValueError("video dimensions and frame rate must be positive")
    return [
        binary,
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]


def encode_mp4(frames: torch.Tensor, output: Path, *, fps: int = 12) -> None:
    """Encode an ``F×H×W×3`` float tensor to an H.264 MP4."""
    if frames.ndim != 4 or frames.shape[-1] != 3 or not len(frames):
        raise ValueError("frames must have shape (F,H,W,3)")
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise RuntimeError("ffmpeg is required to export prompt videos")
    output.parent.mkdir(parents=True, exist_ok=True)
    pixels = (
        frames.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
    )
    process = subprocess.Popen(
        ffmpeg_command(
            binary,
            output,
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
            fps=fps,
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _, stderr = process.communicate(pixels.tobytes())
    if process.returncode:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace').strip()}")


class PromptVideoRuntime:
    """Resident prompt speaker, trajectory realizer, and support renderer."""

    def __init__(
        self,
        paths: PromptVideoPaths,
        *,
        device: str = "cuda:0",
        generation_jewels: int = 72000,
        frames: int = 49,
        height: int = 144,
        width: int = 216,
        fps: int = 12,
        render_batch_frames: int = 4,
    ) -> None:
        if generation_jewels != 72000 or (frames, height, width) != (49, 144, 216):
            raise ValueError("the proof runtime keeps the frozen 72k / 49×144×216 contract")
        if min(fps, render_batch_frames) <= 0:
            raise ValueError("render batch and frame rate must be positive")
        self.paths = paths
        self.device = torch.device(device)
        self.generation_jewels = generation_jewels
        self.frames = frames
        self.height = height
        self.width = width
        self.fps = fps
        self.render_batch_frames = render_batch_frames
        self._clip_model = None
        self._clip_tokenizer = None
        self._load_generation_stack()

    def _load_generation_stack(self) -> None:
        block = torch.load(
            self.paths.block_checkpoint, map_location="cpu", weights_only=False
        )
        architecture = block["architecture"]
        if (
            int(architecture["block_vocabulary_size"]) != 1024
            or tuple(architecture["block_shape"]) != (16, 16, 8)
        ):
            raise ValueError("the demo requires the frozen fine K=1024 language")
        block_codebook = BlockTokenCodebook.from_state_dict(
            block["block_codebook"], self.device
        )
        protocol = json.loads(self.paths.split_report.read_text())["protocol"]
        records = load_field_records([Path(root) for root in protocol["roots"]])
        training, _ = select_prompt_splits(
            records,
            set(protocol["validation_sources"]),
            set(protocol["training_sources"]),
        )
        self.training = sorted(training, key=lambda record: record.source_id)
        if [record.source_id for record in self.training] != list(
            block["training_sources"]
        ):
            raise ValueError("fine language and demo source tokens do not align")
        self.scene_keys = tuple(sorted({scene_key(record) for record in self.training}))
        self.prompts = tuple(key[1] for key in self.scene_keys)
        key_to_scene = {key: index for index, key in enumerate(self.scene_keys)}
        training_scenes = torch.tensor(
            [key_to_scene[scene_key(record)] for record in self.training],
            dtype=torch.long,
            device=self.device,
        )
        self.physical_codebook = load_factorized_codebook(
            self.paths.physical_codebook, self.device
        )
        self.realizer, _ = fit_semantic_trajectory_realizer(
            [record.features.to(self.device) for record in self.training],
            training_scenes,
            block_codebook=block_codebook,
            physical_codebook=self.physical_codebook,
            jitter_std=0.005,
        )
        scene_sources = tuple(
            tuple(torch.nonzero(training_scenes == scene).flatten().cpu().tolist())
            for scene in range(len(self.prompts))
        )
        self.exact_speaker = PromptTrajectorySpeaker(self.prompts, scene_sources)
        self.learned_speaker = None
        self.saved_learned = None
        if self.paths.learned_speaker is not None and self.paths.learned_speaker.exists():
            saved = torch.load(
                self.paths.learned_speaker, map_location="cpu", weights_only=False
            )
            if saved.get("schema") != "learned-trajectory-speaker-checkpoint-v1":
                raise ValueError("unsupported learned speaker checkpoint")
            if list(saved["training_sources"]) != [
                record.source_id for record in self.training
            ]:
                raise ValueError("learned speaker and demo source tokens do not align")
            self.learned_speaker = LearnedTrajectorySpeaker(**saved["model_args"]).to(
                self.device
            ).eval()
            self.learned_speaker.load_state_dict(saved["model"])
            self.saved_learned = saved

    @property
    def learned_available(self) -> bool:
        return self.learned_speaker is not None

    def _learned_text_embedding(self, prompt: str) -> torch.Tensor:
        assert self.saved_learned is not None
        cached = self.saved_learned["prompt_embeddings"].get(prompt)
        if cached is not None:
            return cached.to(self.device)[None]
        if self._clip_model is None:
            import open_clip  # noqa: PLC0415

            self._clip_model = open_clip.create_model(
                "ViT-B-32", pretrained="laion2b_s34b_b79k"
            ).to(self.device).eval()
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = self._clip_tokenizer([prompt]).to(self.device)
        with torch.no_grad():
            return F.normalize(self._clip_model.encode_text(tokens).float(), dim=1)

    @torch.no_grad()
    def generate_field(self, prompt: str, seed: int, *, mode: str) -> GeneratedJewelField:
        """Speak a prompt program and realize it as exactly 72,000 Jewels."""
        prompt = normalize_prompt(prompt)
        if mode == "exact":
            program = self.exact_speaker.compile(prompt, seed)
            program_dict = program.to_dict()
        elif mode == "learned":
            if self.learned_speaker is None:
                raise ValueError("the learned free-form speaker is not installed")
            generator = torch.Generator(device=self.device).manual_seed(seed)
            sampled = self.learned_speaker.sample(
                self._learned_text_embedding(prompt),
                generator=generator,
                temperature=0.8,
                top_k=6,
            )
            program_dict = {
                **asdict(sampled),
                "prompt": prompt,
                "seed": seed,
                "condition": "learned free-form extrapolation",
            }
            program = sampled
        else:
            raise ValueError("mode must be 'exact' or 'learned'")
        field_seed = realization_seed(mode, seed, int(program.scene_token))
        field_generator = torch.Generator(device=self.device).manual_seed(field_seed)
        centers, tokens, realization = self.realizer.sample_rank_balanced_from_donors(
            int(program.scene_token),
            int(program.foreground_token),
            int(program.background_token),
            self.generation_jewels,
            generator=field_generator,
        )
        features = active_tokens_to_features(
            centers, tokens, self.physical_codebook
        )
        background = self.training[int(program.background_token)].background
        metadata = {
            "schema": DEMO_VERSION,
            "prompt": prompt,
            "mode": mode,
            "seed": int(seed),
            "realization_seed": int(field_seed),
            "program": program_dict,
            "program_scene_label": self.scene_keys[int(program.scene_token)][0],
            "foreground_source_id": self.training[
                int(program.foreground_token)
            ].source_id,
            "background_source_id": self.training[
                int(program.background_token)
            ].source_id,
            "realization": realization,
            "render": {
                "frames": self.frames,
                "height": self.height,
                "width": self.width,
                "fps": self.fps,
                "jewels": self.generation_jewels,
            },
            "limitations": (
                "Exact mode is the passing three-prompt proof. Learned mode is a bounded "
                "three-scene extrapolator, not a general open-vocabulary text-to-video model."
            ),
        }
        return GeneratedJewelField(features, background, metadata)

    @torch.no_grad()
    def render_frames(self, field: GeneratedJewelField) -> torch.Tensor:
        """Render the full 49-frame field in bounded point batches."""
        _, features_to_field = _featurizer()
        jewel_field = features_to_field(field.features, device=self.device)
        from stprim.models.render import render_points  # noqa: PLC0415

        rendered = []
        for start in range(0, self.frames, self.render_batch_frames):
            indices = torch.arange(
                start,
                min(start + self.render_batch_frames, self.frames),
                dtype=torch.long,
            )
            points = frame_points(
                self.frames,
                indices,
                self.height,
                self.width,
                device=self.device,
            )
            pixels = render_points(
                jewel_field,
                points,
                cull_mode="support_tiled",
                support_sigma=5.0,
                support_capacity=16384,
                support_point_chunk=min(8192, len(points)),
                support_base_resolution=32,
                support_level_scale=1.55,
                background=field.background.to(points),
            )
            rendered.append(
                pixels.reshape(len(indices), self.height, self.width, 3).clamp(0, 1).cpu()
            )
        return torch.cat(rendered)

    def generate_video(
        self, prompt: str, seed: int, *, mode: str, output_dir: Path
    ) -> tuple[Path, Path, dict[str, Any]]:
        """Generate one field, render it, and save MP4 plus JSON provenance."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = video_basename(prompt, mode, seed)
        video_path = output_dir / f"{stem}.mp4"
        metadata_path = output_dir / f"{stem}.json"
        field = self.generate_field(prompt, seed, mode=mode)
        frames = self.render_frames(field)
        encode_mp4(frames, video_path, fps=self.fps)
        metadata_path.write_text(json.dumps(field.metadata, indent=2) + "\n")
        return video_path, metadata_path, field.metadata
