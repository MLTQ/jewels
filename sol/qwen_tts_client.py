"""Small standard-library client for Pharaoh's asynchronous Qwen3-TTS API."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ProgressCallback = Callable[[str, float], None]


@dataclass(frozen=True)
class QwenCustomVoiceRequest:
    """Typed custom-voice request with deterministic sampling controls."""

    text: str
    speaker: str = "Ryan"
    language: str = "en"
    instruct: str = ""
    seed: int = 0
    temperature: float = 0.5
    top_p: float = 0.85
    max_new_tokens: int = 2048

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Qwen narration text must not be empty")
        if not self.speaker.strip() or not self.language.strip():
            raise ValueError("Qwen speaker and language must not be empty")
        if self.seed < 0 or self.max_new_tokens <= 0:
            raise ValueError("Qwen seed and token limit are invalid")
        if not 0 < self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("Qwen sampling parameters are invalid")

    def payload(self) -> dict[str, Any]:
        """Return the Pharaoh request body, forcing server-local output storage."""
        return {**asdict(self), "output_path": ""}


@dataclass(frozen=True)
class QwenVoiceCloneRequest:
    """Typed reference-conditioned request for a consistent original voice."""

    text: str
    ref_audio_path: str
    ref_transcript: str
    language: str = "en"
    icl_mode: bool = True
    seed: int = 0
    temperature: float = 0.45
    top_p: float = 0.8
    max_new_tokens: int = 768

    def __post_init__(self) -> None:
        if not self.text.strip() or not self.ref_audio_path.strip():
            raise ValueError("Qwen clone text and reference path must not be empty")
        if self.icl_mode and not self.ref_transcript.strip():
            raise ValueError("Qwen ICL cloning requires the reference transcript")
        if self.seed < 0 or self.max_new_tokens <= 0:
            raise ValueError("Qwen seed and token limit are invalid")
        if not 0 < self.temperature <= 2 or not 0 < self.top_p <= 1:
            raise ValueError("Qwen sampling parameters are invalid")

    def payload(self) -> dict[str, Any]:
        return {**asdict(self), "output_path": ""}


class QwenTTSClient:
    """Submit, poll, and download one Pharaoh Qwen3-TTS job at a time."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout: float = 30.0,
        job_timeout: float = 900.0,
        poll_interval: float = 2.0,
    ) -> None:
        normalized = base_url.strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError("Qwen TTS URL must begin with http:// or https://")
        if min(request_timeout, job_timeout, poll_interval) <= 0:
            raise ValueError("Qwen TTS timeouts must be positive")
        self.base_url = normalized
        self.request_timeout = request_timeout
        self.job_timeout = job_timeout
        self.poll_interval = poll_interval

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data is not None else {},
        )
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                body = response.read()
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Qwen TTS HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise ConnectionError(f"Qwen TTS is unreachable at {self.base_url}: {error}") from error
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError("Qwen TTS returned invalid JSON") from error
        if not isinstance(decoded, dict):
            raise RuntimeError("Qwen TTS returned a non-object response")
        return decoded

    def health(self) -> dict[str, Any]:
        health = self._request_json("/health")
        if health.get("status") != "ok" or health.get("stub") is True:
            raise RuntimeError(f"Qwen TTS health check failed: {health}")
        return health

    def upload_file(self, source: Path, *, filename: str | None = None) -> str:
        """Upload one clone reference and return the server-local path."""
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Qwen TTS upload source is unavailable: {source}")
        safe_name = filename or source.name
        if not safe_name or "/" in safe_name or "\\" in safe_name:
            raise ValueError("Qwen TTS upload filename must be a basename")
        request = Request(
            self.base_url + f"/upload?filename={safe_name}",
            data=source.read_bytes(),
            method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                body = response.read()
        except (HTTPError, URLError) as error:
            raise RuntimeError(f"Qwen TTS reference upload failed: {error}") from error
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as error:
            raise RuntimeError("Qwen TTS upload returned invalid JSON") from error
        server_path = decoded.get("server_path") if isinstance(decoded, dict) else None
        if not isinstance(server_path, str) or not server_path:
            raise RuntimeError(f"Qwen TTS upload omitted server_path: {decoded}")
        return server_path

    def _download(self, job_id: str, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_suffix(output.suffix + ".partial")
        request = Request(self.base_url + f"/files/{job_id}", method="GET")
        try:
            with urlopen(request, timeout=self.request_timeout) as response:
                with partial.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            if partial.stat().st_size == 0:
                raise RuntimeError("Qwen TTS downloaded an empty audio file")
            partial.replace(output)
        except (HTTPError, URLError, OSError, RuntimeError) as error:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"Qwen TTS audio download failed: {error}") from error

    def _generate(
        self,
        endpoint: str,
        payload: dict[str, Any],
        output: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> str:
        submitted = self._request_json(
            f"/generate/{endpoint}", method="POST", payload=payload
        )
        job_id = submitted.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise RuntimeError(f"Qwen TTS submission omitted job_id: {submitted}")

        deadline = time.monotonic() + self.job_timeout
        last_progress = -1.0
        while True:
            job = self._request_json(f"/jobs/{job_id}")
            status = str(job.get("status", ""))
            amount = float(job.get("progress") or 0.0)
            if progress is not None and (amount != last_progress or status != "running"):
                progress(status, amount)
                last_progress = amount
            if status == "complete":
                break
            if status == "failed":
                raise RuntimeError(f"Qwen TTS job {job_id} failed: {job.get('error')}")
            if status not in {"pending", "running"}:
                raise RuntimeError(f"Qwen TTS job {job_id} has unknown status {status!r}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Qwen TTS job {job_id} exceeded {self.job_timeout}s")
            time.sleep(self.poll_interval)

        self._download(job_id, output)
        return job_id

    def generate_custom_voice(
        self,
        request: QwenCustomVoiceRequest,
        output: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Block on one asynchronous CustomVoice job and download its WAV."""
        job_id = self._generate(
            "custom_voice", request.payload(), output, progress=progress
        )
        return {
            "job_id": job_id,
            "speaker": request.speaker,
            "language": request.language,
            "instruct": request.instruct,
            "seed": request.seed,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_new_tokens": request.max_new_tokens,
        }

    def generate_voice_clone(
        self,
        request: QwenVoiceCloneRequest,
        output: Path,
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Block on one reference-conditioned clone job and download its WAV."""
        job_id = self._generate("voice_clone", request.payload(), output, progress=progress)
        return {
            "job_id": job_id,
            "language": request.language,
            "icl_mode": request.icl_mode,
            "seed": request.seed,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "max_new_tokens": request.max_new_tokens,
        }
