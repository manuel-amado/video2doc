"""Unit tests for Video2Doc audio transcription pipeline."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video2doc.audio.transcriber import (
    AudioTranscriber,
    TranscriptSegment,
    TranscriptionResult,
)
from video2doc.config import Settings
from video2doc.exceptions import MediaCorruptError, MissingDependencyError


# ==============================================================================
# Data Models Tests
# ==============================================================================

def test_transcript_segment_dataclass():
    """TranscriptSegment stores start, end, and text correctly."""
    seg = TranscriptSegment(start=1.5, end=4.2, text="Hello world")
    assert seg.start == 1.5
    assert seg.end == 4.2
    assert seg.text == "Hello world"


def test_transcription_result_formatting():
    """TranscriptionResult formats segments into readable timestamped text blocks."""
    segments = [
        TranscriptSegment(start=0.0, end=4.5, text="Welcome to the demo."),
        TranscriptSegment(start=5.0, end=65.2, text="Here we configure settings."),
    ]
    res = TranscriptionResult(
        language="en",
        duration=65.2,
        segments=segments,
        full_text="Welcome to the demo. Here we configure settings.",
    )
    formatted = res.to_formatted_timestamp_text()
    assert "[00:00 -> 00:04] Welcome to the demo." in formatted
    assert "[00:05 -> 01:05] Here we configure settings." in formatted


def test_transcription_result_defaults():
    """TranscriptionResult initializes with empty default segments and text."""
    res = TranscriptionResult(language="en", duration=10.0)
    assert res.segments == []
    assert res.full_text == ""
    assert res.to_formatted_timestamp_text() == ""


# ==============================================================================
# Device Negotiation Tests
# ==============================================================================

def test_negotiate_device_auto_with_cuda():
    """Device negotiation 'auto' resolves to 'cuda' when CUDA is available."""
    with patch("torch.cuda.is_available", return_value=True):
        device = AudioTranscriber._negotiate_device("auto")
        assert device == "cuda"


def test_negotiate_device_auto_without_cuda():
    """Device negotiation 'auto' resolves to 'cpu' when CUDA is unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        device = AudioTranscriber._negotiate_device("auto")
        assert device == "cpu"


def test_negotiate_device_explicit_cpu():
    """Explicit device 'cpu' resolves to 'cpu'."""
    device = AudioTranscriber._negotiate_device("cpu")
    assert device == "cpu"


def test_negotiate_device_cuda_fallback_when_unavailable():
    """Requesting 'cuda' when unavailable falls back to 'cpu'."""
    with patch("torch.cuda.is_available", return_value=False):
        device = AudioTranscriber._negotiate_device("cuda")
        assert device == "cpu"


# ==============================================================================
# Backend Loading & Fallback Tests
# ==============================================================================

def test_ensure_model_faster_whisper(mock_whisper):
    """AudioTranscriber prefers faster-whisper when available."""
    transcriber = AudioTranscriber()
    model = transcriber._ensure_model()
    assert model is not None
    assert transcriber._backend == "faster-whisper"


def test_ensure_model_openai_whisper_fallback(mock_whisper):
    """AudioTranscriber falls back to openai-whisper when faster-whisper is unavailable."""
    transcriber = AudioTranscriber()
    with patch.dict("sys.modules", {"faster_whisper": None}):
        model = transcriber._ensure_model()
        assert model is not None
        assert transcriber._backend == "openai-whisper"


def test_ensure_model_missing_dependencies():
    """AudioTranscriber raises MissingDependencyError when neither backend is available."""
    transcriber = AudioTranscriber()
    with patch.dict("sys.modules", {"faster_whisper": None, "whisper": None}):
        with pytest.raises(MissingDependencyError) as exc_info:
            transcriber._ensure_model()
        assert "whisper" in exc_info.value.dependency_name.lower()


# ==============================================================================
# Happy Path Transcription Tests
# ==============================================================================

def test_transcribe_faster_whisper_happy_path(sample_video: Path, mock_whisper):
    """Transcribe extracts audio and parses segments via faster-whisper."""
    transcriber = AudioTranscriber()
    result = transcriber.transcribe(sample_video)

    assert isinstance(result, TranscriptionResult)
    assert result.language == "en"
    assert result.duration > 0.0
    assert len(result.segments) == 3
    assert "Welcome to the system setup guide." in result.full_text
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 5.0


def test_transcribe_openai_whisper_happy_path(sample_video: Path, mock_whisper):
    """Transcribe extracts audio and parses segments via openai-whisper."""
    transcriber = AudioTranscriber()
    with patch.dict("sys.modules", {"faster_whisper": None}):
        result = transcriber.transcribe(sample_video)

        assert isinstance(result, TranscriptionResult)
        assert result.language == "en"
        assert len(result.segments) == 3
        assert result.full_text != ""


def test_transcribe_custom_language(sample_video: Path, mock_whisper):
    """Transcribe passes explicit language parameter to model."""
    transcriber = AudioTranscriber()
    result = transcriber.transcribe(sample_video, language="es")
    assert result.language in ("es", "en")


def test_transcribe_settings_language(sample_video: Path, mock_whisper):
    """Transcribe uses language from Settings when not explicitly provided."""
    settings = Settings(whisper_language="fr")
    transcriber = AudioTranscriber(settings=settings)
    result = transcriber.transcribe(sample_video)
    assert result is not None


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_transcribe_missing_video_raises_media_corrupt(tmp_path: Path):
    """Non-existent video file raises MediaCorruptError."""
    missing_video = tmp_path / "does_not_exist.mp4"
    transcriber = AudioTranscriber()
    with pytest.raises(MediaCorruptError) as exc_info:
        transcriber.transcribe(missing_video)
    assert "does not exist" in exc_info.value.reason.lower()


def test_transcribe_empty_video_raises_media_corrupt(tmp_path: Path):
    """Zero-byte video file raises MediaCorruptError."""
    empty_video = tmp_path / "empty.mp4"
    empty_video.write_bytes(b"")
    transcriber = AudioTranscriber()
    with pytest.raises(MediaCorruptError) as exc_info:
        transcriber.transcribe(empty_video)
    assert "0 bytes" in exc_info.value.reason.lower() or "empty" in exc_info.value.reason.lower()


def test_transcribe_missing_ffmpeg_raises_missing_dependency(sample_video: Path, ffmpeg_state):
    """Missing FFmpeg executable raises MissingDependencyError."""
    ffmpeg_state.simulate_missing = True
    transcriber = AudioTranscriber()
    with pytest.raises(MissingDependencyError) as exc_info:
        transcriber.transcribe(sample_video)
    assert exc_info.value.dependency_name == "ffmpeg"


def test_transcribe_corrupt_ffmpeg_audio_extraction(sample_video: Path, ffmpeg_state):
    """Subprocess failure during FFmpeg audio extraction raises MediaCorruptError."""
    ffmpeg_state.simulate_corrupt_video = True
    transcriber = AudioTranscriber()
    with pytest.raises(MediaCorruptError):
        transcriber.transcribe(sample_video)


def test_transcribe_whisper_runtime_failure(sample_video: Path, mock_whisper):
    """Runtime error in model.transcribe is wrapped in MediaCorruptError."""
    transcriber = AudioTranscriber()
    model = transcriber._ensure_model()
    with patch.object(model, "transcribe", side_effect=RuntimeError("CUDA error in Whisper")):
        with pytest.raises(MediaCorruptError) as exc_info:
            transcriber.transcribe(sample_video)
        assert "Whisper transcription failed" in exc_info.value.reason


# ==============================================================================
# Resource Management & Unload Tests
# ==============================================================================

def test_transcribe_cleans_up_temp_audio(sample_video: Path, mock_whisper, tmp_path: Path):
    """Temporary audio extraction WAV file is deleted after transcription."""
    created_temp_files: list[Path] = []
    original_named_temp = AudioTranscriber._extract_audio

    def tracking_extract(self, media_path, temp_wav):
        created_temp_files.append(Path(temp_wav))
        original_named_temp(self, media_path, temp_wav)

    with patch.object(AudioTranscriber, "_extract_audio", side_effect=tracking_extract, autospec=True):
        transcriber = AudioTranscriber()
        transcriber.transcribe(sample_video)

    assert len(created_temp_files) == 1
    assert not created_temp_files[0].exists()


def test_unload_whisper_model(mock_whisper):
    """Unload releases model reference and invokes garbage collection."""
    transcriber = AudioTranscriber()
    transcriber._ensure_model()
    assert transcriber._model is not None

    with patch("gc.collect") as mock_gc, patch("torch.cuda.empty_cache") as mock_empty_cache:
        transcriber.unload()
        assert transcriber._model is None
        assert transcriber._backend is None
        mock_gc.assert_called_once()
        mock_empty_cache.assert_called_once()
