"""Global pytest fixtures and synthetic mocking infrastructure for Video2Doc E2E tests.

Provides deterministic, offline mocks for:
- FFmpeg and ffprobe subprocess calls
- Ollama HTTP API (/api/chat, /api/generate, /api/tags, model unloads)
- Whisper and faster-whisper ASR backends
- PyTorch CUDA memory management
- CLI test runner
"""

import base64
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

# Minimal valid 1x1 JPEG bytes for realistic keyframe mocking
TINY_JPEG_BYTES = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00"
    b"\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19"
    b"\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82"
    b"<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00"
    b"\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04"
    b"\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
)


@dataclass
class OllamaCallRecord:
    endpoint: str
    model: Optional[str]
    prompt: Optional[str]
    keep_alive: Optional[Any]
    has_images: bool
    payload: Dict[str, Any]


class MockOllamaState:
    """Tracks calls to Ollama and allows simulating error states."""

    def __init__(self) -> None:
        self.calls: List[OllamaCallRecord] = []
        self.unloaded_models: List[str] = []
        self.simulate_down: bool = False
        self.simulate_missing_model: Optional[str] = None
        self.simulate_timeout: bool = False
        self.available_models = [
            "llama3.2-vision:latest",
            "llama3.2-vision",
            "qwen2.5-coder:14b:latest",
            "qwen2.5-coder:14b",
            "llava:latest",
            "llava",
        ]

    def record_call(self, endpoint: str, payload: Dict[str, Any]) -> None:
        model = payload.get("model")
        prompt = payload.get("prompt")
        if not prompt and "messages" in payload:
            messages = payload.get("messages", [])
            prompt = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict))
        keep_alive = payload.get("keep_alive")
        has_images = bool(payload.get("images")) or any(
            bool(m.get("images")) for m in payload.get("messages", []) if isinstance(m, dict)
        )
        if keep_alive == 0 or keep_alive == "0":
            if model:
                self.unloaded_models.append(model)
        self.calls.append(
            OllamaCallRecord(
                endpoint=endpoint,
                model=model,
                prompt=prompt,
                keep_alive=keep_alive,
                has_images=has_images,
                payload=payload,
            )
        )


class MockFFmpegState:
    """Tracks calls to FFmpeg/ffprobe and allows simulating errors."""

    def __init__(self) -> None:
        self.calls: List[List[str]] = []
        self.simulate_missing: bool = False
        self.simulate_corrupt_video: bool = False
        self.mock_duration: float = 60.0
        self.keyframes_created: List[Path] = []


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provides a fresh Typer CliRunner for opaque-box CLI tests."""
    return CliRunner()


@pytest.fixture
def ollama_state() -> MockOllamaState:
    return MockOllamaState()


@pytest.fixture
def ffmpeg_state() -> MockFFmpegState:
    return MockFFmpegState()


@pytest.fixture
def mock_cuda():
    """Mocks torch.cuda methods to verify VRAM cache flushes without hardware GPU."""
    empty_cache_mock = MagicMock()
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.empty_cache", empty_cache_mock
    ):
        yield {"empty_cache": empty_cache_mock}


@pytest.fixture
def mock_ollama(ollama_state: MockOllamaState):
    """Mocks HTTP requests to Ollama API (/api/chat, /api/generate, /api/tags, /api/version)."""

    def fake_post(url: str, json: Optional[Dict[str, Any]] = None, **kwargs: Any):
        if ollama_state.simulate_down:
            import requests

            raise requests.exceptions.ConnectionError("Failed to connect to Ollama server.")

        if ollama_state.simulate_timeout:
            import requests

            raise requests.exceptions.Timeout("Ollama request timed out.")

        json = json or {}
        model = json.get("model", "")
        ollama_state.record_call(url, json)

        if ollama_state.simulate_missing_model and model == ollama_state.simulate_missing_model:
            resp = MagicMock()
            resp.status_code = 404
            resp.json.return_value = {"error": f"model '{model}' not found"}
            resp.text = f"model '{model}' not found"
            resp.raise_for_status.side_effect = Exception(f"Model {model} not found")
            return resp

        # Model unload signal
        if json.get("keep_alive") == 0 or json.get("keep_alive") == "0":
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {"status": "success", "done": True}
            resp.text = '{"status": "success"}'
            return resp

        # Vision description call
        has_images = bool(json.get("images")) or any(
            bool(m.get("images")) for m in json.get("messages", []) if isinstance(m, dict)
        )
        if has_images or "vision" in model.lower() or "llava" in model.lower():
            desc = (
                "Keyframe visual analysis: Trading platform dashboard showing active strategy "
                "parameters, moving average indicators, and an execution confirmation dialog."
            )
            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "model": model,
                "message": {"role": "assistant", "content": desc},
                "response": desc,
                "done": True,
            }
            resp.text = desc
            return resp

        # Text synthesis call
        doc_content = (
            "# Generated Documentation\n\n"
            "## 1. Overview\n"
            "Comprehensive procedural documentation generated from multimodal analysis.\n\n"
            "## 2. Step-by-Step Instructions\n"
            "1. Launch the platform interface.\n"
            "2. Navigate to Strategy Configuration.\n"
            "3. Verify parameter thresholds and click Execute.\n\n"
            "## 3. Visual Reference Summary\n"
            "- Screenshot confirms parameters window is configured properly.\n\n"
            "## 4. Key Takeaways & Action Items\n"
            "- Parameter verification complete.\n"
        )
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "model": model,
            "message": {"role": "assistant", "content": doc_content},
            "response": doc_content,
            "done": True,
        }
        resp.text = doc_content
        return resp

    def fake_get(url: str, **kwargs: Any):
        if ollama_state.simulate_down:
            import requests

            raise requests.exceptions.ConnectionError("Ollama daemon unreachable.")

        resp = MagicMock()
        resp.status_code = 200
        if url.endswith("/api/tags") or "/api/tags" in url:
            resp.json.return_value = {
                "models": [{"name": m} for m in ollama_state.available_models]
            }
        else:
            resp.json.return_value = {"version": "0.3.14"}
        return resp

    with patch("requests.post", side_effect=fake_post), patch(
        "requests.get", side_effect=fake_get
    ):
        yield ollama_state


@pytest.fixture
def mock_ffmpeg(ffmpeg_state: MockFFmpegState):
    """Mocks FFmpeg and ffprobe subprocess commands and binary discovery."""

    orig_which = shutil.which
    orig_run = subprocess.run
    orig_check_output = subprocess.check_output

    def fake_which(cmd: str, *args: Any, **kwargs: Any) -> Optional[str]:
        if ffmpeg_state.simulate_missing and ("ffmpeg" in cmd or "ffprobe" in cmd):
            return None
        if "ffmpeg" in cmd:
            return "/usr/bin/ffmpeg" if os.name != "nt" else r"C:\ffmpeg\bin\ffmpeg.exe"
        if "ffprobe" in cmd:
            return "/usr/bin/ffprobe" if os.name != "nt" else r"C:\ffmpeg\bin\ffprobe.exe"
        return orig_which(cmd, *args, **kwargs)

    def fake_run(cmd, *args: Any, **kwargs: Any):
        cmd_list = [str(c) for c in cmd] if isinstance(cmd, (list, tuple)) else str(cmd).split()
        if not any("ffmpeg" in str(c).lower() or "ffprobe" in str(c).lower() for c in cmd_list):
            return orig_run(cmd, *args, **kwargs)
        ffmpeg_state.calls.append(cmd_list)

        if ffmpeg_state.simulate_corrupt_video:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd_list, stderr=b"Invalid data found when processing input"
            )

        # Detect keyframe extraction output path
        out_path_str = cmd_list[-1]
        if out_path_str.endswith(".jpg") or out_path_str.endswith(".png"):
            out_path = Path(out_path_str)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(TINY_JPEG_BYTES)
            ffmpeg_state.keyframes_created.append(out_path)

        res = MagicMock(spec=subprocess.CompletedProcess)
        res.returncode = 0
        res.stdout = b"ffmpeg version 6.0 mock"
        res.stderr = b""
        return res

    def fake_check_output(cmd, *args: Any, **kwargs: Any):
        cmd_list = [str(c) for c in cmd] if isinstance(cmd, (list, tuple)) else str(cmd).split()
        if not any("ffmpeg" in str(c).lower() or "ffprobe" in str(c).lower() for c in cmd_list):
            return orig_check_output(cmd, *args, **kwargs)
        ffmpeg_state.calls.append(cmd_list)

        if ffmpeg_state.simulate_corrupt_video:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd_list, stderr=b"Invalid media file header"
            )

        if "format=duration" in " ".join(cmd_list):
            return f"{ffmpeg_state.mock_duration}\n".encode()

        return b"60.0\n"

    with patch("shutil.which", side_effect=fake_which), patch(
        "subprocess.run", side_effect=fake_run
    ), patch("subprocess.check_output", side_effect=fake_check_output):
        yield ffmpeg_state


@pytest.fixture
def mock_whisper():
    """Mocks Whisper and faster-whisper ASR backends to return realistic transcription segments."""
    from collections import namedtuple

    Segment = namedtuple("Segment", ["start", "end", "text"])

    fake_segments = [
        Segment(0.0, 5.0, "Welcome to the system setup guide."),
        Segment(5.0, 12.5, "First open the configuration panel and choose parameters."),
        Segment(12.5, 20.0, "Next click the run button to execute the pipeline."),
    ]

    class FakeFasterWhisperModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.model_size_or_path = args[0] if args else kwargs.get("model_size_or_path", "base")

        def transcribe(self, audio: Any, **kwargs: Any):
            Info = namedtuple("Info", ["language", "language_probability", "duration"])
            info = Info(language="en", language_probability=0.99, duration=20.0)
            return (fake_segments, info)

    class FakeOpenAIWhisperModel:
        def transcribe(self, audio: Any, **kwargs: Any):
            return {
                "language": "en",
                "segments": [
                    {"start": s.start, "end": s.end, "text": s.text} for s in fake_segments
                ],
                "text": " ".join(s.text for s in fake_segments),
            }

    with patch("whisper.load_model", return_value=FakeOpenAIWhisperModel()), patch(
        "faster_whisper.WhisperModel", FakeFasterWhisperModel
    ):
        yield {
            "segments": fake_segments,
            "faster_model": FakeFasterWhisperModel,
            "openai_model": FakeOpenAIWhisperModel,
        }


@pytest.fixture(autouse=True)
def mock_environment(mock_ffmpeg, mock_ollama, mock_whisper, mock_cuda):
    """Default active environment fixture ensuring all E2E tests run offline deterministically."""
    return {
        "ffmpeg": mock_ffmpeg,
        "ollama": mock_ollama,
        "whisper": mock_whisper,
        "cuda": mock_cuda,
    }


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Creates a temporary dummy video file."""
    video = tmp_path / "sample_video.mp4"
    video.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41")
    return video


@pytest.fixture
def sample_video_batch(tmp_path: Path) -> Path:
    """Creates a directory with multiple dummy video files for batch processing tests."""
    batch_dir = tmp_path / "batch_videos"
    batch_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 4):
        v = batch_dir / f"tutorial_module_{i}.mp4"
        v.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41")
    return batch_dir


@pytest.fixture
def sample_custom_template(tmp_path: Path) -> Path:
    """Creates a valid user-defined custom YAML template."""
    tmpl_dir = tmp_path / "templates"
    tmpl_dir.mkdir(parents=True, exist_ok=True)
    tmpl_file = tmpl_dir / "custom_workflow.yaml"
    content = """name: custom_workflow
description: Custom engineering workflow template
vision_prompt: >
  Analyze the engineering diagram or UI screenshot.
  Extract component names, connection arrows, and status badges.
synthesis_prompt: >
  # {{ title }}
  
  ## Architecture & Workflow
  {{ visual_summary }}
  
  ## Transcription Notes
  {{ transcription_text }}
  
  ## Engineering Checklist
  {% for item in checklist %}
  - [ ] {{ item }}
  {% endfor %}
sections:
  - Architecture & Workflow
  - Transcription Notes
  - Engineering Checklist
"""
    tmpl_file.write_text(content, encoding="utf-8")
    return tmpl_file
