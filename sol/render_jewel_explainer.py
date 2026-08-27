"""Render the six-part narrated mathematical Jewel explainer series."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable

from PIL import Image

from sol.jewel_explainer_episodes import EPISODES, Episode, Shot, episode_by_number
from sol.jewel_explainer_scenes import SCENE_RENDERERS, draw_shot
from sol.jewel_explainer_style import BACKGROUND, HEIGHT, WIDTH
from sol.qwen_tts_client import (
    QwenCustomVoiceRequest,
    QwenTTSClient,
    QwenVoiceCloneRequest,
)


DEFAULT_OUTPUT = Path("sol/results/jewel_explainer_series_v1")
DEFAULT_QWEN_REFERENCE = DEFAULT_OUTPUT / "officer_voice_reference.wav"
DEFAULT_QWEN_REFERENCE_TRANSCRIPT = (
    "The current result is a bounded existence proof, not a general text-to-video model. "
    "The important point is causal direction: text becomes a program, the program becomes a "
    "continuous spacetime field, and the field becomes pixels."
)
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
DEFAULT_QWEN_INSTRUCT = (
    "Speak like an exceptional mathematical lecturer: calm, precise, intellectually curious, "
    "natural conversational cadence, measured emphasis, no announcer voice, and brief pauses "
    "around technical clauses."
)


@dataclass(frozen=True)
class NarrationConfig:
    """Complete and auditable settings for one narration backend."""

    backend: str
    say_voice: str
    say_rate: int
    qwen_url: str
    qwen_speaker: str
    qwen_instruct: str
    qwen_reference_audio: Path
    qwen_reference_transcript: str
    qwen_seed: int
    qwen_temperature: float
    qwen_top_p: float
    qwen_max_new_tokens: int
    qwen_request_timeout: float
    qwen_job_timeout: float
    qwen_poll_interval: float
    qwen_max_attempts: int

    def validate(self) -> None:
        if self.backend not in {"say", "qwen-custom", "qwen-clone"}:
            raise ValueError("narration backend must be say, qwen-custom, or qwen-clone")
        if self.say_rate <= 0 or self.qwen_seed < 0:
            raise ValueError("narration rate and seed are invalid")
        if not 0 < self.qwen_temperature <= 2 or not 0 < self.qwen_top_p <= 1:
            raise ValueError("Qwen narration sampling is invalid")
        if self.qwen_max_new_tokens <= 0 or self.qwen_max_attempts <= 0:
            raise ValueError("Qwen narration token ceiling and attempts must be positive")
        if self.backend == "qwen-clone":
            if not self.qwen_reference_audio.is_file():
                raise FileNotFoundError(
                    f"Qwen clone reference is unavailable: {self.qwen_reference_audio}"
                )
            if not self.qwen_reference_transcript.strip():
                raise ValueError("Qwen clone reference transcript must not be empty")


def narration_duration_bounds(text: str) -> tuple[float, float, float]:
    """Estimate expected, minimum, and maximum speech seconds from prose length."""
    word_count = len(re.findall(r"\b[\w'-]+\b", text))
    expected = max(5.0, word_count * 60.0 / 145.0)
    minimum = max(5.0, expected * 0.60)
    maximum = min(75.0, max(20.0, expected * 1.50 + 4.0))
    return expected, minimum, maximum


def qwen_token_ceiling(text: str, hard_ceiling: int) -> int:
    """Translate the duration guard into Qwen's approximately 12.5-Hz token budget."""
    if hard_ceiling <= 0:
        raise ValueError("Qwen hard token ceiling must be positive")
    _, _, maximum = narration_duration_bounds(text)
    return min(hard_ceiling, math.ceil(maximum * 12.5))


def run(command: list[str]) -> None:
    """Run one artifact command and surface concise failure output."""
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"command failed ({command[0]}): {detail}")


def probe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    duration = float(completed.stdout.strip())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid media duration for {path}")
    return duration


def format_srt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def subtitle_rows(
    shots: Iterable[Shot], durations: Iterable[float], *, tail_seconds: float
) -> list[tuple[float, float, str]]:
    """Distribute each shot's sentences over its spoken interval."""
    rows = []
    offset = 0.0
    for shot, duration in zip(shots, durations):
        spoken = max(0.1, duration - tail_seconds)
        sentences = [row.strip() for row in SENTENCE_BOUNDARY.split(shot.narration) if row.strip()]
        weights = [max(1, len(sentence)) for sentence in sentences]
        total = sum(weights)
        cursor = offset
        for sentence, weight in zip(sentences, weights):
            allocation = spoken * weight / total
            rows.append((cursor, cursor + allocation, sentence))
            cursor += allocation
        offset += duration
    return rows


def write_srt(path: Path, rows: list[tuple[float, float, str]]) -> None:
    blocks = []
    for index, (start, end, text) in enumerate(rows, 1):
        blocks.append(
            f"{index}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n"
        )
    path.write_text("\n".join(blocks), encoding="utf-8")


def synthesize_qwen_shot(
    client: QwenTTSClient,
    episode: Episode,
    shot: Shot,
    shot_index: int,
    work_dir: Path,
    config: NarrationConfig,
    *,
    reference_server_path: str | None,
) -> tuple[Path, dict[str, Any]]:
    """Generate one bounded take, retrying temporal outliers with new deterministic seeds."""
    expected, minimum, maximum = narration_duration_bounds(shot.narration)
    token_ceiling = qwen_token_ceiling(shot.narration, config.qwen_max_new_tokens)
    base_seed = config.qwen_seed + episode.number * 100 + shot_index
    rejected: list[dict[str, Any]] = []
    for attempt in range(config.qwen_max_attempts):
        seed = base_seed + attempt * 10_000
        source = work_dir / f"shot_{shot_index:02d}.attempt_{attempt + 1}.wav"
        if config.backend == "qwen-clone":
            if reference_server_path is None:
                raise RuntimeError("Qwen clone reference was not uploaded")
            request = QwenVoiceCloneRequest(
                text=shot.narration,
                ref_audio_path=reference_server_path,
                ref_transcript=config.qwen_reference_transcript,
                language="en",
                icl_mode=True,
                seed=seed,
                temperature=config.qwen_temperature,
                top_p=config.qwen_top_p,
                max_new_tokens=token_ceiling,
            )
            client.generate_voice_clone(request, source)
        else:
            request = QwenCustomVoiceRequest(
                text=shot.narration,
                speaker=config.qwen_speaker,
                language="en",
                instruct=config.qwen_instruct,
                seed=seed,
                temperature=config.qwen_temperature,
                top_p=config.qwen_top_p,
                max_new_tokens=token_ceiling,
            )
            client.generate_custom_voice(request, source)
        duration = probe_duration(source)
        accepted = minimum <= duration <= maximum and duration < maximum * 0.97
        attempt_record = {
            "attempt": attempt + 1,
            "seed": seed,
            "duration_seconds": duration,
        }
        if accepted:
            return source, {
                "expected_seconds": expected,
                "minimum_seconds": minimum,
                "maximum_seconds": maximum,
                "token_ceiling": token_ceiling,
                "accepted": attempt_record,
                "rejected": rejected,
            }
        rejected.append(attempt_record)
        print(
            f"rejected episode {episode.number} shot {shot_index + 1} Qwen take "
            f"{attempt + 1}: {duration:.1f}s outside {minimum:.1f}-{maximum:.1f}s",
            flush=True,
        )
    raise RuntimeError(
        f"episode {episode.number} shot {shot_index + 1} produced no bounded Qwen take: "
        f"{rejected}"
    )


def synthesize_narration(
    episode: Episode,
    work_dir: Path,
    *,
    config: NarrationConfig,
    tail_seconds: float,
) -> tuple[list[Path], list[float], dict[str, Any]]:
    """Create one padded WAV per shot so animation timing follows real speech."""
    config.validate()
    paths = []
    durations = []
    qwen_client = None
    qwen_health: dict[str, Any] = {}
    reference_server_path = None
    reference_sha256 = None
    qwen_shots: list[dict[str, Any]] = []
    if config.backend.startswith("qwen-"):
        qwen_client = QwenTTSClient(
            config.qwen_url,
            request_timeout=config.qwen_request_timeout,
            job_timeout=config.qwen_job_timeout,
            poll_interval=config.qwen_poll_interval,
        )
        qwen_health = qwen_client.health()
        if config.backend == "qwen-clone":
            reference_sha256 = hashlib.sha256(
                config.qwen_reference_audio.read_bytes()
            ).hexdigest()
            reference_server_path = qwen_client.upload_file(
                config.qwen_reference_audio,
                filename=f"jewel-officer-{reference_sha256[:16]}.wav",
            )
    for index, shot in enumerate(episode.shots):
        wav = work_dir / f"shot_{index:02d}.wav"
        if qwen_client is not None:
            source, diagnostics = synthesize_qwen_shot(
                qwen_client,
                episode,
                shot,
                index,
                work_dir,
                config,
                reference_server_path=reference_server_path,
            )
            qwen_shots.append({"shot": index + 1, **diagnostics})
        else:
            source = work_dir / f"shot_{index:02d}.aiff"
            text_path = work_dir / f"shot_{index:02d}.txt"
            text_path.write_text(shot.narration, encoding="utf-8")
            run([
                "say",
                "-v",
                config.say_voice,
                "-r",
                str(config.say_rate),
                "-o",
                str(source),
                "-f",
                str(text_path),
            ])
        run([
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            f"loudnorm=I=-18:TP=-1.5:LRA=11,apad=pad_dur={tail_seconds}",
            "-ar",
            "48000",
            "-ac",
            "1",
            str(wav),
        ])
        paths.append(wav)
        durations.append(probe_duration(wav))
        print(
            f"narrated episode {episode.number} shot {index + 1}/{len(episode.shots)} "
            f"with {config.backend} "
            f"({durations[-1]:.1f}s)",
            flush=True,
        )
    if qwen_client is not None:
        qwen_health = qwen_client.health()
        metadata = {
            "backend": f"pharaoh-qwen3-tts-{config.backend.removeprefix('qwen-')}",
            "service_url": config.qwen_url,
            "service_model": qwen_health.get("model_variant"),
            "speaker": (
                "original-warm-american-first-officer"
                if config.backend == "qwen-clone"
                else config.qwen_speaker
            ),
            "language": "en",
            "instruct": config.qwen_instruct if config.backend == "qwen-custom" else None,
            "seed_rule": (
                f"{config.qwen_seed} + episode * 100 + shot_index + retry * 10000"
            ),
            "temperature": config.qwen_temperature,
            "top_p": config.qwen_top_p,
            "hard_max_new_tokens": config.qwen_max_new_tokens,
            "max_attempts": config.qwen_max_attempts,
            "loudness_target_lufs": -18,
            "duration_guard": "145 WPM estimate; accept 0.60x to 1.50x+4s; reject cap hits",
            "shot_generation": qwen_shots,
        }
        if config.backend == "qwen-clone":
            metadata.update({
                "reference_audio": config.qwen_reference_audio.name,
                "reference_sha256": reference_sha256,
                "reference_transcript": config.qwen_reference_transcript,
                "icl_mode": True,
            })
    else:
        metadata = {
            "backend": "macos-say",
            "speaker": config.say_voice,
            "speech_rate": config.say_rate,
            "loudness_target_lufs": -18,
        }
    return paths, durations, metadata


def concat_audio(paths: list[Path], output: Path, work_dir: Path) -> None:
    manifest = work_dir / "audio_concat.txt"
    manifest.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in paths), encoding="utf-8"
    )
    run([
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest),
        "-c:a",
        "pcm_s16le",
        str(output),
    ])


def decode_video_frames(path: Path) -> list[Image.Image]:
    """Decode a retained proof MP4 to small RGB frames for exact playback inserts."""
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, height = (int(value) for value in completed.stdout.strip().split("x"))
    decoded = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    frame_bytes = width * height * 3
    if len(decoded) % frame_bytes:
        raise ValueError(f"decoded byte count does not align for {path}")
    return [
        Image.frombytes("RGB", (width, height), decoded[offset : offset + frame_bytes])
        for offset in range(0, len(decoded), frame_bytes)
    ]


def focus_evidence_asset(name: str, source: Image.Image) -> Image.Image:
    """Reflow tall audit sheets into legible, claim-specific horizontal montages."""
    image = source.convert("RGB")
    if name == "coherent":
        scene_height = image.height // 3
        pair_height = round(scene_height * 2 / 5)
        montage = Image.new("RGB", (image.width * 3, pair_height), BACKGROUND)
        for scene in range(3):
            top = scene * scene_height
            crop = image.crop((0, top, image.width, top + pair_height))
            montage.paste(crop, (scene * image.width, 0))
        return montage
    if name == "proof-sheet":
        left = round(image.width * 0.277)
        top = round(image.height * 0.041)
        row_step = round(image.height * 0.160)
        row_height = round(image.height * 0.154)
        crop_width = image.width - left
        montage = Image.new("RGB", (crop_width * 3, row_height), BACKGROUND)
        for row in range(3):
            crop_top = top + row * row_step
            crop = image.crop(
                (left, crop_top, image.width, crop_top + row_height)
            )
            montage.paste(crop, (row * crop_width, 0))
        return montage
    return image


def load_assets(project_root: Path) -> dict[str, Any]:
    result = project_root / "sol" / "results" / "jewel_casting_language_v0"
    assets: dict[str, Any] = {}
    image_paths = {
        "coherent": result / "coherent_source_oracle_v1" / "qualitative_source_disjoint.png",
        "proof-sheet": result / "trajectory_speaker_evidence_v1" / "trajectory_speaker_proof_sheet.png",
        "evidence": result / "trajectory_speaker_evidence_v1" / "trajectory_speaker_evidence.png",
    }
    for name, path in image_paths.items():
        if path.exists():
            assets[name] = focus_evidence_asset(name, Image.open(path))
    videos = project_root / "sol" / "results" / "jewel_prompt_demo_v1" / "generated"
    clip_paths = {
        "ballerina": next(iter(sorted(videos.glob("a-ballerina*.mp4"))), None),
        "dog": next(iter(sorted(videos.glob("a-golden-retriever*.mp4"))), None),
        "welder": next(iter(sorted(videos.glob("a-welder*.mp4"))), None),
    }
    for name, path in clip_paths.items():
        if path is not None:
            assets[f"clip:{name}"] = decode_video_frames(path)
    isolation_path = (
        project_root
        / "sol"
        / "results"
        / "jewel_explainer_series_v1"
        / "assets"
        / "actual_jewel_isolation.mp4"
    )
    if isolation_path.exists():
        assets["clip:actual-jewels"] = decode_video_frames(isolation_path)
    return assets


def render_silent_video(
    episode: Episode,
    durations: list[float],
    assets: dict[str, Any],
    output: Path,
    *,
    fps: int,
) -> int:
    total_frames = sum(max(1, round(duration * fps)) for duration in durations)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{WIDTH}x{HEIGHT}",
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
        "17",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    emitted = 0
    try:
        for shot_index, (shot, duration) in enumerate(zip(episode.shots, durations)):
            frames = max(1, round(duration * fps))
            for frame_index in range(frames):
                shot_progress = frame_index / max(frames - 1, 1)
                episode_progress = (emitted + frame_index) / max(total_frames - 1, 1)
                image = draw_shot(
                    episode, shot, shot_progress, episode_progress, assets
                )
                process.stdin.write(image.tobytes())
            emitted += frames
            print(
                f"rendered episode {episode.number} shot {shot_index + 1}/{len(episode.shots)}",
                flush=True,
            )
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
    except Exception:
        process.kill()
        process.wait()
        raise
    if return_code:
        raise RuntimeError(f"video encoder failed: {stderr.decode(errors='replace')}")
    return emitted


def mux_episode(video: Path, audio: Path, subtitles: Path, output: Path) -> None:
    partial = output.with_suffix(".partial.mp4")
    run([
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-i",
        str(audio),
        "-i",
        str(subtitles),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:s:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-c:s",
        "mov_text",
        "-metadata:s:s:0",
        "language=eng",
        "-shortest",
        "-movflags",
        "+faststart",
        str(partial),
    ])
    partial.replace(output)


def write_episode_script(episode: Episode, output: Path) -> None:
    rows = [
        f"# Episode {episode.number}: {episode.title}",
        "",
        episode.subtitle,
        "",
        "## Claim sources",
        "",
        *(f"- `{source}`" for source in episode.sources),
        "",
    ]
    for index, shot in enumerate(episode.shots, 1):
        rows.extend([
            f"## {index}. {shot.title}",
            "",
            shot.narration,
            "",
            f"**On screen:** {shot.caption}",
            "",
        ])
    output.write_text("\n".join(rows), encoding="utf-8")


def render_episode(
    episode: Episode,
    output_dir: Path,
    assets: dict[str, Any],
    *,
    fps: int,
    narration_config: NarrationConfig,
    tail_seconds: float,
) -> dict[str, Any]:
    stem = f"episode_{episode.number:02d}_{episode.slug}"
    video_path = output_dir / f"{stem}.mp4"
    subtitle_path = output_dir / f"{stem}.srt"
    script_path = output_dir / f"{stem}_script.md"
    poster_path = output_dir / f"{stem}_poster.png"
    with tempfile.TemporaryDirectory(prefix=f"{stem}_", dir=output_dir) as temporary:
        work_dir = Path(temporary)
        audio_paths, durations, narration_metadata = synthesize_narration(
            episode,
            work_dir,
            config=narration_config,
            tail_seconds=tail_seconds,
        )
        narration = work_dir / "narration.wav"
        silent_video = work_dir / "silent.mp4"
        concat_audio(audio_paths, narration, work_dir)
        write_srt(
            subtitle_path,
            subtitle_rows(episode.shots, durations, tail_seconds=tail_seconds),
        )
        frames = render_silent_video(
            episode, durations, assets, silent_video, fps=fps
        )
        mux_episode(silent_video, narration, subtitle_path, video_path)
    write_episode_script(episode, script_path)
    draw_shot(episode, episode.shots[0], 0.82, 0.02, assets).save(poster_path)
    duration = probe_duration(video_path)
    return {
        "episode": episode.number,
        "slug": episode.slug,
        "title": episode.title,
        "video": video_path.name,
        "poster": poster_path.name,
        "subtitles": subtitle_path.name,
        "script": script_path.name,
        "duration_seconds": duration,
        "frames": frames,
        "fps": fps,
        "resolution": [WIDTH, HEIGHT],
        "voice": narration_metadata["speaker"],
        "speech_rate": narration_metadata.get("speech_rate"),
        "narration": narration_metadata,
        "sources": list(episode.sources),
    }


def write_contact_sheet(records: list[dict[str, Any]], output: Path) -> None:
    posters = [
        Image.open(output.parent / record["poster"]).convert("RGB") for record in records
    ]
    thumb_width, thumb_height = 480, 270
    sheet = Image.new("RGB", (thumb_width * 2, thumb_height * 3), "#0b1020")
    for index, poster in enumerate(posters):
        poster.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(poster, ((index % 2) * thumb_width, (index // 2) * thumb_height))
    sheet.save(output)


def merge_episode_records(
    previous: list[dict[str, Any]],
    rendered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace rerendered episodes while preserving the rest of an artifact set."""
    merged = {int(record["episode"]): record for record in previous}
    merged.update({int(record["episode"]): record for record in rendered})
    return [merged[number] for number in sorted(merged)]


def validate_specs(project_root: Path) -> None:
    numbers = [episode.number for episode in EPISODES]
    slugs = [episode.slug for episode in EPISODES]
    if numbers != list(range(1, 7)) or len(slugs) != len(set(slugs)):
        raise ValueError("series requires six uniquely named episodes numbered 1 through 6")
    for episode in EPISODES:
        if len(episode.shots) != 7:
            raise ValueError(f"episode {episode.number} must contain seven shots")
        if episode.theme not in {"dark", "light"}:
            raise ValueError(
                f"episode {episode.number} has unknown theme {episode.theme!r}"
            )
        for source in episode.sources:
            if not (project_root / source).exists():
                raise FileNotFoundError(f"episode source does not exist: {source}")
        for shot in episode.shots:
            if shot.visual not in SCENE_RENDERERS:
                raise ValueError(f"unregistered visual {shot.visual}")
            if min(len(shot.narration), len(shot.caption)) < 20:
                raise ValueError("every shot needs substantive narration and caption")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode", action="append", type=int, default=[])
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument(
        "--tts-backend",
        choices=("qwen-clone", "qwen-custom", "qwen", "say"),
        default="qwen-clone",
    )
    parser.add_argument("--voice", default="Daniel")
    parser.add_argument("--speech-rate", type=int, default=175)
    parser.add_argument(
        "--qwen-url",
        default=os.environ.get("JEWEL_QWEN_TTS_URL", "http://192.168.0.202:18001"),
    )
    parser.add_argument("--qwen-speaker", default="Ryan")
    parser.add_argument("--qwen-instruct", default=DEFAULT_QWEN_INSTRUCT)
    parser.add_argument("--qwen-reference-audio", type=Path, default=DEFAULT_QWEN_REFERENCE)
    parser.add_argument(
        "--qwen-reference-transcript", default=DEFAULT_QWEN_REFERENCE_TRANSCRIPT
    )
    parser.add_argument("--qwen-seed", type=int, default=20261001)
    parser.add_argument("--qwen-temperature", type=float, default=0.45)
    parser.add_argument("--qwen-top-p", type=float, default=0.8)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=900)
    parser.add_argument("--qwen-max-attempts", type=int, default=3)
    parser.add_argument("--qwen-request-timeout", type=float, default=30.0)
    parser.add_argument("--qwen-job-timeout", type=float, default=900.0)
    parser.add_argument("--qwen-poll-interval", type=float, default=2.0)
    parser.add_argument("--tail-seconds", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.speech_rate <= 0 or args.tail_seconds < 0:
        raise ValueError("render timing arguments are invalid")
    project_root = args.project_root.resolve()
    reference_audio = args.qwen_reference_audio
    if not reference_audio.is_absolute():
        reference_audio = project_root / reference_audio
    narration_config = NarrationConfig(
        backend="qwen-custom" if args.tts_backend == "qwen" else args.tts_backend,
        say_voice=args.voice,
        say_rate=args.speech_rate,
        qwen_url=args.qwen_url,
        qwen_speaker=args.qwen_speaker,
        qwen_instruct=args.qwen_instruct,
        qwen_reference_audio=reference_audio,
        qwen_reference_transcript=args.qwen_reference_transcript,
        qwen_seed=args.qwen_seed,
        qwen_temperature=args.qwen_temperature,
        qwen_top_p=args.qwen_top_p,
        qwen_max_new_tokens=args.qwen_max_new_tokens,
        qwen_request_timeout=args.qwen_request_timeout,
        qwen_job_timeout=args.qwen_job_timeout,
        qwen_poll_interval=args.qwen_poll_interval,
        qwen_max_attempts=args.qwen_max_attempts,
    )
    narration_config.validate()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_specs(project_root)
    selected = (
        [episode_by_number(number) for number in args.episode]
        if args.episode
        else list(EPISODES)
    )
    assets = load_assets(project_root)
    records = []
    for episode in selected:
        records.append(
            render_episode(
                episode,
                output_dir,
                assets,
                fps=args.fps,
                narration_config=narration_config,
                tail_seconds=args.tail_seconds,
            )
        )
        print(f"completed episode {episode.number}: {records[-1]['video']}", flush=True)
    inventory_path = output_dir / "inventory.json"
    if args.episode and inventory_path.exists():
        previous_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        if previous_inventory.get("schema") != "jewel-technical-explainer-series-v1":
            raise ValueError("existing output inventory has an incompatible schema")
        records = merge_episode_records(
            list(previous_inventory.get("episodes", [])), records
        )
    inventory = {
        "schema": "jewel-technical-explainer-series-v1",
        "episodes": records,
        "render_contract": {
            "resolution": [WIDTH, HEIGHT],
            "fps": args.fps,
            "codec": "H.264 yuv420p + AAC + mov_text subtitles",
            "narration_backend": narration_config.backend,
            "voice": (
                "original-warm-american-first-officer"
                if narration_config.backend == "qwen-clone"
                else narration_config.qwen_speaker
                if narration_config.backend == "qwen-custom"
                else narration_config.say_voice
            ),
            "speech_rate": (
                narration_config.say_rate if narration_config.backend == "say" else None
            ),
            "style": "original mathematical vector animation; episode 2 uses an eggshell light palette",
        },
    }
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    if len(records) == 6:
        write_contact_sheet(records, output_dir / "series_contact_sheet.png")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
