"""Unit tests for DocumentPipeline (multi-stage video documentation orchestrator)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from video2doc.audio.transcriber import TranscriptionResult, TranscriptSegment
from video2doc.config import Settings
from video2doc.exceptions import (
    MediaCorruptError,
    ModelNotFoundError,
    OllamaConnectionError,
    Video2DocError,
)
from video2doc.pipeline import DocumentPipeline
from video2doc.vision.keyframes import Keyframe


@pytest.fixture
def dummy_video(tmp_path: Path) -> Path:
    """Create a non-empty mock video file."""
    vpath = tmp_path / "tutorial_video.mp4"
    vpath.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41")
    return vpath


@pytest.fixture
def dummy_empty_video(tmp_path: Path) -> Path:
    """Create an empty 0-byte video file."""
    vpath = tmp_path / "zero_bytes.mp4"
    vpath.write_bytes(b"")
    return vpath


class TestDocumentPipelineInit:
    """Tests for DocumentPipeline component initialization and defaults."""

    def test_default_initialization(self):
        pipeline = DocumentPipeline()
        assert pipeline.settings is not None
        assert pipeline.audio_transcriber is not None
        assert pipeline.keyframe_extractor is not None
        assert pipeline.vision_analyzer is not None
        assert pipeline.synthesizer is not None
        assert pipeline.memory_manager is not None
        assert pipeline.template_engine is not None

    def test_custom_settings_initialization(self):
        settings = Settings(
            vision_model="llava:7b",
            text_model="llama3.1:8b",
            output_dir=Path("custom_output"),
        )
        pipeline = DocumentPipeline(settings=settings)
        assert pipeline.settings.vision_model == "llava:7b"
        assert pipeline.settings.text_model == "llama3.1:8b"
        assert pipeline.vision_analyzer.model_name == "llava:7b"
        assert pipeline.synthesizer.model_name == "llama3.1:8b"


class TestDiagnostics:
    """Tests for pre-flight health and model verification methods."""

    def test_check_ollama_health_success(self):
        pipeline = DocumentPipeline()
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200

        with patch("requests.get", return_value=mock_resp):
            assert pipeline.check_ollama_health() is True

    def test_check_ollama_health_connection_error(self):
        pipeline = DocumentPipeline()
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Connection refused")):
            with pytest.raises(OllamaConnectionError):
                pipeline.check_ollama_health()

    def test_check_ollama_health_http_failure(self):
        pipeline = DocumentPipeline()
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 503

        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(OllamaConnectionError):
                pipeline.check_ollama_health()

    def test_validate_models_available_all_present(self):
        settings = Settings(vision_model="llama3.2-vision", text_model="qwen2.5-coder:14b")
        pipeline = DocumentPipeline(settings=settings)

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3.2-vision:latest"},
                {"name": "qwen2.5-coder:14b"},
            ]
        }

        with patch("requests.get", return_value=mock_resp):
            assert pipeline.validate_models_available() is True

    def test_validate_models_available_missing_vision_model(self):
        settings = Settings(vision_model="missing-vision", text_model="qwen2.5-coder:14b")
        pipeline = DocumentPipeline(settings=settings)

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:14b"},
            ]
        }

        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(ModelNotFoundError) as exc_info:
                pipeline.validate_models_available()
            assert "missing-vision" in exc_info.value.model_name

    def test_validate_models_available_missing_text_model(self):
        settings = Settings(vision_model="llama3.2-vision", text_model="missing-text")
        pipeline = DocumentPipeline(settings=settings)

        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3.2-vision:latest"},
            ]
        }

        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(ModelNotFoundError) as exc_info:
                pipeline.validate_models_available()
            assert "missing-text" in exc_info.value.model_name


class TestProcessVideo:
    """Tests for sequential 5-stage pipeline processing."""

    def test_process_video_nonexistent_raises_media_corrupt(self, tmp_path: Path):
        pipeline = DocumentPipeline()
        missing = tmp_path / "does_not_exist.mp4"
        with pytest.raises(MediaCorruptError):
            pipeline.process_video(missing)

    def test_process_video_empty_file_raises_media_corrupt(self, dummy_empty_video: Path):
        pipeline = DocumentPipeline()
        with pytest.raises(MediaCorruptError):
            pipeline.process_video(dummy_empty_video)

    def test_process_video_sequential_execution_happy_path(self, dummy_video: Path, tmp_path: Path):
        pipeline = DocumentPipeline()
        out_dir = tmp_path / "pipeline_out"

        fake_trans = TranscriptionResult(
            language="en",
            duration=30.0,
            segments=[TranscriptSegment(0.0, 10.0, "Transcribed step one.")],
            full_text="Transcribed step one.",
        )
        fake_frame = tmp_path / "frame_0001.jpg"
        fake_frame.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
        fake_keyframes = [Keyframe(timestamp=5.0, frame_path=fake_frame, score=0.5)]
        fake_visual_descriptions = [
            {"timestamp": 5.0, "frame_path": fake_frame, "description": "UI panel", "score": 0.5}
        ]
        fake_markdown = "# Generated SOP\n\n## Overview\nComplete instructions."

        stage_cleanup_calls: list[tuple[str, str | None]] = []

        def track_stage_cleanup(stage_name: str, model_name: str | None = None):
            stage_cleanup_calls.append((stage_name, model_name))

        with patch.object(pipeline.audio_transcriber, "transcribe", return_value=fake_trans), patch.object(
            pipeline.audio_transcriber, "unload"
        ) as mock_asr_unload, patch.object(
            pipeline.keyframe_extractor, "extract", return_value=fake_keyframes
        ), patch.object(
            pipeline.vision_analyzer, "analyze_keyframes", return_value=fake_visual_descriptions
        ), patch.object(
            pipeline.synthesizer, "generate_document", return_value=fake_markdown
        ), patch.object(
            pipeline.memory_manager, "stage_cleanup", side_effect=track_stage_cleanup
        ):
            doc_path = pipeline.process_video(dummy_video, template="sop", output_dir=out_dir)

            assert doc_path.exists()
            assert doc_path.name == f"{dummy_video.stem}.md"
            content = doc_path.read_text(encoding="utf-8")
            assert len(content) > 0

            # Verify staged unload and cleanup sequence
            mock_asr_unload.assert_called_once()
            assert ("audio", None) in stage_cleanup_calls
            assert ("vision", pipeline.settings.vision_model) in stage_cleanup_calls
            assert ("synthesis", pipeline.settings.text_model) in stage_cleanup_calls

    def test_process_video_silent_video_handling(self, dummy_video: Path, tmp_path: Path):
        pipeline = DocumentPipeline()
        out_dir = tmp_path / "silent_out"

        empty_trans = TranscriptionResult(language="en", duration=15.0, segments=[], full_text="")

        with patch.object(pipeline.audio_transcriber, "transcribe", return_value=empty_trans), patch.object(
            pipeline.keyframe_extractor, "extract", return_value=[]
        ), patch.object(
            pipeline.vision_analyzer, "analyze_keyframes", return_value=[]
        ), patch.object(
            pipeline.synthesizer, "generate_document", return_value="# Silent Video Doc\n\nVisual overview."
        ):
            doc_path = pipeline.process_video(dummy_video, template="lecture", output_dir=out_dir)
            assert doc_path.exists()
            content = doc_path.read_text(encoding="utf-8")
            assert len(content) > 0
            assert "#" in content


class TestProcessBatch:
    """Tests for process_batch directory traversal and error isolation."""

    def test_process_batch_invalid_directory_raises_video2doc_error(self, tmp_path: Path):
        pipeline = DocumentPipeline()
        invalid_dir = tmp_path / "nonexistent_dir"
        with pytest.raises(Video2DocError):
            pipeline.process_batch(invalid_dir)

    def test_process_batch_empty_directory_returns_empty_list(self, tmp_path: Path):
        pipeline = DocumentPipeline()
        empty_dir = tmp_path / "empty_batch"
        empty_dir.mkdir(parents=True, exist_ok=True)
        results = pipeline.process_batch(empty_dir)
        assert results == []

    def test_process_batch_multiple_videos(self, tmp_path: Path):
        pipeline = DocumentPipeline()
        batch_dir = tmp_path / "batch_input"
        batch_dir.mkdir(parents=True, exist_ok=True)

        v1 = batch_dir / "mod_01.mp4"
        v2 = batch_dir / "mod_02.mkv"
        v3 = batch_dir / "mod_03.webm"
        txt_ignore = batch_dir / "readme.txt"

        for v in (v1, v2, v3):
            v.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41")
        txt_ignore.write_text("ignore me", encoding="utf-8")

        out_dir = tmp_path / "batch_out"

        with patch.object(pipeline, "process_video") as mock_proc:
            mock_proc.side_effect = [
                out_dir / "mod_01.md",
                out_dir / "mod_02.md",
                out_dir / "mod_03.md",
            ]

            results = pipeline.process_batch(batch_dir, template="sop", output_dir=out_dir)
            assert len(results) == 3
            assert mock_proc.call_count == 3

    def test_process_batch_error_isolation(self, tmp_path: Path):
        """Verify that a corrupt or failing video does not abort subsequent video processing."""
        pipeline = DocumentPipeline()
        batch_dir = tmp_path / "failing_batch"
        batch_dir.mkdir(parents=True, exist_ok=True)

        v1 = batch_dir / "good_1.mp4"
        v2 = batch_dir / "bad_2.mp4"
        v3 = batch_dir / "good_3.mp4"

        for v in (v1, v2, v3):
            v.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

        out_dir = tmp_path / "failing_out"

        def fake_process_video(video_path=None, *args, **kwargs):
            vpath = video_path if video_path is not None else (args[0] if args else kwargs.get("video_path"))
            if "bad_2" in str(vpath):
                raise MediaCorruptError(str(vpath), reason="Corrupt media stream")
            return out_dir / f"{Path(vpath).stem}.md"

        with patch.object(pipeline, "process_video", side_effect=fake_process_video):
            results = pipeline.process_batch(batch_dir, output_dir=out_dir)
            assert len(results) == 2
            assert any("good_1" in str(p) for p in results)
            assert any("good_3" in str(p) for p in results)
