"""Tests for the Pharaoh Qwen3-TTS request, polling, and download contracts."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sol.qwen_tts_client import (
    QwenCustomVoiceRequest,
    QwenTTSClient,
    QwenVoiceCloneRequest,
)


class FakeResponse:
    def __init__(self, payload: dict | bytes) -> None:
        encoded = json.dumps(payload).encode("utf-8") if isinstance(payload, dict) else payload
        self.buffer = BytesIO(encoded)

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.buffer.read(size)


class QwenTTSClientTests(unittest.TestCase):
    def test_request_validation_and_remote_output_payload(self) -> None:
        request = QwenCustomVoiceRequest(text="A Jewel is a Gaussian in spacetime.", seed=7)
        self.assertEqual(request.payload()["output_path"], "")
        self.assertEqual(request.payload()["speaker"], "Ryan")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            QwenCustomVoiceRequest(text="  ")
        with self.assertRaisesRegex(ValueError, "sampling"):
            QwenCustomVoiceRequest(text="valid", top_p=0)

    def test_submit_poll_and_atomic_download(self) -> None:
        captured_payload: dict = {}
        poll_count = 0

        def fake_urlopen(request: object, *, timeout: float) -> FakeResponse:
            nonlocal poll_count
            self.assertEqual(timeout, 3.0)
            url = request.full_url  # type: ignore[attr-defined]
            if url.endswith("/health"):
                return FakeResponse({"status": "ok", "stub": False})
            if url.endswith("/generate/custom_voice"):
                captured_payload.update(json.loads(request.data))  # type: ignore[attr-defined]
                return FakeResponse({"job_id": "job-1"})
            if url.endswith("/jobs/job-1"):
                poll_count += 1
                if poll_count == 1:
                    return FakeResponse({"status": "running", "progress": 0.4})
                return FakeResponse({"status": "complete", "progress": 1.0})
            if url.endswith("/files/job-1"):
                return FakeResponse(b"RIFF-qwen-test-audio")
            raise AssertionError(f"unexpected URL {url}")

        client = QwenTTSClient(
            "http://tts.test/", request_timeout=3.0, job_timeout=1.0, poll_interval=0.001
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "line.wav"
            with patch("sol.qwen_tts_client.urlopen", side_effect=fake_urlopen):
                self.assertEqual(client.health()["status"], "ok")
                metadata = client.generate_custom_voice(
                    QwenCustomVoiceRequest(text="Technical narration.", seed=42), output
                )
            self.assertEqual(output.read_bytes(), b"RIFF-qwen-test-audio")
            self.assertFalse(output.with_suffix(".wav.partial").exists())
        self.assertEqual(metadata["job_id"], "job-1")
        self.assertEqual(captured_payload["seed"], 42)
        self.assertEqual(captured_payload["output_path"], "")
        self.assertEqual(poll_count, 2)

    def test_reference_upload_and_icl_clone_payload(self) -> None:
        captured_upload = b""
        captured_clone: dict = {}

        def fake_urlopen(request: object, *, timeout: float) -> FakeResponse:
            nonlocal captured_upload
            self.assertEqual(timeout, 2.0)
            url = request.full_url  # type: ignore[attr-defined]
            if "/upload?filename=voice.wav" in url:
                captured_upload = request.data  # type: ignore[attr-defined]
                return FakeResponse({"server_path": "/server/voice.wav"})
            if url.endswith("/generate/voice_clone"):
                captured_clone.update(json.loads(request.data))  # type: ignore[attr-defined]
                return FakeResponse({"job_id": "clone-1"})
            if url.endswith("/jobs/clone-1"):
                return FakeResponse({"status": "complete", "progress": 1.0})
            if url.endswith("/files/clone-1"):
                return FakeResponse(b"RIFF-cloned-officer")
            raise AssertionError(f"unexpected URL {url}")

        client = QwenTTSClient(
            "http://tts.test", request_timeout=2.0, job_timeout=1.0, poll_interval=0.001
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.wav"
            reference.write_bytes(b"RIFF-reference")
            output = root / "clone.wav"
            with patch("sol.qwen_tts_client.urlopen", side_effect=fake_urlopen):
                server_path = client.upload_file(reference, filename="voice.wav")
                metadata = client.generate_voice_clone(
                    QwenVoiceCloneRequest(
                        text="New technical line.",
                        ref_audio_path=server_path,
                        ref_transcript="Reference technical line.",
                        seed=99,
                    ),
                    output,
                )
            self.assertEqual(output.read_bytes(), b"RIFF-cloned-officer")
        self.assertEqual(captured_upload, b"RIFF-reference")
        self.assertEqual(captured_clone["ref_audio_path"], "/server/voice.wav")
        self.assertEqual(captured_clone["ref_transcript"], "Reference technical line.")
        self.assertTrue(captured_clone["icl_mode"])
        self.assertEqual(captured_clone["output_path"], "")
        self.assertEqual(metadata["seed"], 99)

    def test_failed_job_surfaces_server_error(self) -> None:
        responses = iter([
            FakeResponse({"job_id": "bad-job"}),
            FakeResponse({"status": "failed", "progress": 1.0, "error": "model error"}),
        ])
        client = QwenTTSClient(
            "http://tts.test", request_timeout=1.0, job_timeout=1.0, poll_interval=0.001
        )
        with patch("sol.qwen_tts_client.urlopen", side_effect=lambda *args, **kwargs: next(responses)):
            with self.assertRaisesRegex(RuntimeError, "model error"):
                client.generate_custom_voice(
                    QwenCustomVoiceRequest(text="Technical narration."), Path("unused.wav")
                )

    def test_clone_requires_transcript_in_icl_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires the reference transcript"):
            QwenVoiceCloneRequest(
                text="Technical narration.",
                ref_audio_path="/server/reference.wav",
                ref_transcript="",
            )


if __name__ == "__main__":
    unittest.main()
