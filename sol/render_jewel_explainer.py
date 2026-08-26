"""Render the six-part narrated mathematical Jewel explainer series."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable

from PIL import Image

from sol.jewel_explainer_episodes import EPISODES, Episode, Shot, episode_by_number
from sol.jewel_explainer_scenes import SCENE_RENDERERS, draw_shot
from sol.jewel_explainer_style import HEIGHT, WIDTH


DEFAULT_OUTPUT = Path("sol/results/jewel_explainer_series_v1")
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


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


def synthesize_narration(
    episode: Episode,
    work_dir: Path,
    *,
    voice: str,
    rate: int,
    tail_seconds: float,
) -> tuple[list[Path], list[float]]:
    """Create one padded WAV per shot so animation timing follows real speech."""
    paths = []
    durations = []
    for index, shot in enumerate(episode.shots):
        aiff = work_dir / f"shot_{index:02d}.aiff"
        wav = work_dir / f"shot_{index:02d}.wav"
        text_path = work_dir / f"shot_{index:02d}.txt"
        text_path.write_text(shot.narration, encoding="utf-8")
        run(["say", "-v", voice, "-r", str(rate), "-o", str(aiff), "-f", str(text_path)])
        run([
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(aiff),
            "-af",
            f"apad=pad_dur={tail_seconds}",
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
            f"({durations[-1]:.1f}s)",
            flush=True,
        )
    return paths, durations


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
            assets[name] = Image.open(path).convert("RGB")
    videos = project_root / "sol" / "results" / "jewel_prompt_demo_v1" / "generated"
    clip_paths = {
        "ballerina": next(iter(sorted(videos.glob("a-ballerina*.mp4"))), None),
        "dog": next(iter(sorted(videos.glob("a-golden-retriever*.mp4"))), None),
        "welder": next(iter(sorted(videos.glob("a-welder*.mp4"))), None),
    }
    for name, path in clip_paths.items():
        if path is not None:
            assets[f"clip:{name}"] = decode_video_frames(path)
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
    voice: str,
    rate: int,
    tail_seconds: float,
) -> dict[str, Any]:
    stem = f"episode_{episode.number:02d}_{episode.slug}"
    video_path = output_dir / f"{stem}.mp4"
    subtitle_path = output_dir / f"{stem}.srt"
    script_path = output_dir / f"{stem}_script.md"
    poster_path = output_dir / f"{stem}_poster.png"
    with tempfile.TemporaryDirectory(prefix=f"{stem}_", dir=output_dir) as temporary:
        work_dir = Path(temporary)
        audio_paths, durations = synthesize_narration(
            episode,
            work_dir,
            voice=voice,
            rate=rate,
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
        "voice": voice,
        "speech_rate": rate,
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


def validate_specs(project_root: Path) -> None:
    numbers = [episode.number for episode in EPISODES]
    slugs = [episode.slug for episode in EPISODES]
    if numbers != list(range(1, 7)) or len(slugs) != len(set(slugs)):
        raise ValueError("series requires six uniquely named episodes numbered 1 through 6")
    for episode in EPISODES:
        if len(episode.shots) != 7:
            raise ValueError(f"episode {episode.number} must contain seven shots")
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
    parser.add_argument("--voice", default="Daniel")
    parser.add_argument("--speech-rate", type=int, default=175)
    parser.add_argument("--tail-seconds", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.speech_rate <= 0 or args.tail_seconds < 0:
        raise ValueError("render timing arguments are invalid")
    project_root = args.project_root.resolve()
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
                voice=args.voice,
                rate=args.speech_rate,
                tail_seconds=args.tail_seconds,
            )
        )
        print(f"completed episode {episode.number}: {records[-1]['video']}", flush=True)
    inventory_path = output_dir / "inventory.json"
    inventory = {
        "schema": "jewel-technical-explainer-series-v1",
        "episodes": records,
        "render_contract": {
            "resolution": [WIDTH, HEIGHT],
            "fps": args.fps,
            "codec": "H.264 yuv420p + AAC + mov_text subtitles",
            "voice": args.voice,
            "speech_rate": args.speech_rate,
            "style": "original dark mathematical vector animation",
        },
    }
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    if len(records) == 6:
        write_contact_sheet(records, output_dir / "series_contact_sheet.png")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
