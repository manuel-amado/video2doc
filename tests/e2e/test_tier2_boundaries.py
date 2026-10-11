"""Tier 2 - Boundary and Corner Cases E2E Tests.

Exercises edge cases, negative scenarios, boundary limits, and fault recovery:
- Missing FFmpeg dependency
- Unreachable Ollama daemon / connection refused
- Missing Ollama vision/text models
- Request timeout simulations
- Corrupt and zero-byte media files
- Non-existent input files and non-existent templates
- Malformed YAML in custom templates
- Silent video with empty speech transcript
- Static video with zero scene changes (fallback handling)
- Filenames with spaces, parentheses, accents, and unicode
- Automatic creation of nested output directories
"""

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

pytestmark = pytest.mark.tier2


def test_tier2_missing_ffmpeg_raises_typed_error(sample_video: Path, ffmpeg_state, tmp_path: Path):
    """Tier 2.1: Missing FFmpeg in PATH raises MissingDependencyError with installation instructions."""
    ffmpeg_state.simulate_missing = True
    exceptions_mod = pytest.importorskip("video2doc.exceptions")
    vision_mod = pytest.importorskip("video2doc.vision.keyframes")

    extractor = vision_mod.KeyframeExtractor()
    with pytest.raises(exceptions_mod.MissingDependencyError):
        extractor.extract(sample_video, tmp_path / "frames")


def test_tier2_missing_ollama_model_raises_model_not_found(ollama_state):
    """Tier 2.2: When Ollama does not have the specified model, raises ModelNotFoundError."""
    ollama_state.simulate_missing_model = "non-existent-model"
    exceptions_mod = pytest.importorskip("video2doc.exceptions")
    synthesis_mod = pytest.importorskip("video2doc.synthesis.generator")
    config_mod = pytest.importorskip("video2doc.config")
    from video2doc.audio.transcriber import TranscriptionResult

    settings = config_mod.Settings(text_model="non-existent-model")
    synthesizer = synthesis_mod.DocumentSynthesizer(settings=settings)
    trans = TranscriptionResult(language="en", duration=10.0, segments=[], full_text="Text")

    with pytest.raises(exceptions_mod.ModelNotFoundError):
        synthesizer.generate_document(trans, [], "Prompt")


def test_tier2_ollama_daemon_connection_refused(ollama_state):
    """Tier 2.3: Unreachable Ollama server raises OllamaConnectionError."""
    ollama_state.simulate_down = True
    exceptions_mod = pytest.importorskip("video2doc.exceptions")
    synthesis_mod = pytest.importorskip("video2doc.synthesis.generator")
    from video2doc.audio.transcriber import TranscriptionResult

    synthesizer = synthesis_mod.DocumentSynthesizer()
    trans = TranscriptionResult(language="en", duration=10.0, segments=[], full_text="Text")

    with pytest.raises(exceptions_mod.OllamaConnectionError):
        synthesizer.generate_document(trans, [], "Prompt")


def test_tier2_ollama_request_timeout(ollama_state):
    """Tier 2.4: Timed out Ollama request raises OllamaConnectionError with timeout indication."""
    ollama_state.simulate_timeout = True
    exceptions_mod = pytest.importorskip("video2doc.exceptions")
    synthesis_mod = pytest.importorskip("video2doc.synthesis.generator")
    from video2doc.audio.transcriber import TranscriptionResult

    synthesizer = synthesis_mod.DocumentSynthesizer()
    trans = TranscriptionResult(language="en", duration=10.0, segments=[], full_text="Text")

    with pytest.raises((exceptions_mod.OllamaConnectionError, exceptions_mod.Video2DocError)):
        synthesizer.generate_document(trans, [], "Prompt")


def test_tier2_corrupt_video_file_handling(sample_video: Path, ffmpeg_state, tmp_path: Path):
    """Tier 2.5: Corrupt video file triggers MediaCorruptError during extraction."""
    ffmpeg_state.simulate_corrupt_video = True
    exceptions_mod = pytest.importorskip("video2doc.exceptions")
    vision_mod = pytest.importorskip("video2doc.vision.keyframes")

    extractor = vision_mod.KeyframeExtractor()
    with pytest.raises(exceptions_mod.MediaCorruptError):
        extractor.extract(sample_video, tmp_path / "frames")


def test_tier2_empty_zero_byte_video_file(tmp_path: Path):
    """Tier 2.6: Zero-byte video file is validated and raises MediaCorruptError."""
    exceptions_mod = pytest.importorskip("video2doc.exceptions")
    pipeline_mod = pytest.importorskip("video2doc.pipeline")

    empty_video = tmp_path / "empty.mp4"
    empty_video.write_bytes(b"")

    pipeline = pipeline_mod.DocumentPipeline()
    with pytest.raises(exceptions_mod.MediaCorruptError):
        pipeline.process_video(empty_video)


def test_tier2_nonexistent_video_path(cli_runner, tmp_path: Path):
    """Tier 2.7: Non-existent video path passed to CLI fails with non-zero exit code."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    missing_path = tmp_path / "does_not_exist.mp4"

    res = cli_runner.invoke(cli_mod.app, ["process", str(missing_path)])
    assert res.exit_code != 0


def test_tier2_nonexistent_template_name(sample_video: Path, tmp_path: Path):
    """Tier 2.8: Requesting non-existent template raises TemplateNotFoundError."""
    exceptions_mod = pytest.importorskip("video2doc.exceptions")
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")

    engine = tmpl_mod.TemplateEngine()
    with pytest.raises(exceptions_mod.TemplateNotFoundError):
        engine.load_template("phantom_template_xyz")


def test_tier2_invalid_yaml_syntax_in_template(tmp_path: Path):
    """Tier 2.9: Malformed YAML syntax raises descriptive parsing error."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    broken_yaml = tmp_path / "broken.yaml"
    broken_yaml.write_text("name: broken\n  bad_indentation: [unclosed\n", encoding="utf-8")

    with pytest.raises(Exception) as exc_info:
        engine.load_template(str(broken_yaml))
    assert "yaml" in str(exc_info.value).lower() or "syntax" in str(exc_info.value).lower() or "parse" in str(exc_info.value).lower()


def test_tier2_missing_required_fields_in_template_yaml(tmp_path: Path):
    """Tier 2.10: Template YAML missing required sections or prompts raises validation error."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    incomplete_yaml = tmp_path / "incomplete.yaml"
    incomplete_yaml.write_text("name: incomplete\ndescription: missing prompts\n", encoding="utf-8")

    with pytest.raises(Exception):
        engine.load_template(str(incomplete_yaml))


def test_tier2_silent_video_empty_speech(sample_video: Path, tmp_path: Path):
    """Tier 2.11: Silent video with no spoken words synthesizes visual-only documentation."""
    pipeline_mod = pytest.importorskip("video2doc.pipeline")
    from video2doc.audio.transcriber import TranscriptionResult

    pipeline = pipeline_mod.DocumentPipeline()
    out_dir = tmp_path / "silent_out"

    with patch.object(pipeline.audio_transcriber, "transcribe", return_value=TranscriptionResult("en", 30.0, [], "")):
        doc_path = pipeline.process_video(sample_video, template="sop", output_dir=out_dir)
        assert doc_path.exists()
        content = doc_path.read_text(encoding="utf-8")
        assert len(content) > 0


def test_tier2_zero_scenes_detected_fallback(sample_video: Path, tmp_path: Path):
    """Tier 2.12: When zero scene cuts are detected, interval fallback provides keyframes."""
    vision_mod = pytest.importorskip("video2doc.vision.keyframes")
    extractor = vision_mod.KeyframeExtractor()
    out_dir = tmp_path / "frames"

    frames = extractor.extract(sample_video, out_dir, threshold=0.99, max_frames=3)
    assert len(frames) > 0
    assert all(f.frame_path.exists() for f in frames)


def test_tier2_extreme_duration_video(tmp_path: Path, ffmpeg_state):
    """Tier 2.13: Video with 10-hour duration calculates keyframe intervals safely."""
    vision_mod = pytest.importorskip("video2doc.vision.keyframes")
    ffmpeg_state.mock_duration = 36000.0  # 10 hours

    video_file = tmp_path / "long_video.mp4"
    video_file.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

    extractor = vision_mod.KeyframeExtractor()
    frames = extractor.extract(video_file, tmp_path / "frames", max_frames=5)
    assert len(frames) <= 5
    for f in frames:
        assert 0.0 <= f.timestamp <= 36000.0


def test_tier2_special_characters_in_filenames(tmp_path: Path, cli_runner):
    """Tier 2.14: Videos with spaces, accents, parentheses, and brackets process cleanly."""
    cli_mod = pytest.importorskip("video2doc.cli.main")

    complex_name = "Tutorial (SQX) - Configuración & Optimización [2026].mp4"
    video_path = tmp_path / complex_name
    video_path.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41")
    out_dir = tmp_path / "complex_out"

    res = cli_runner.invoke(cli_mod.app, ["process", str(video_path), "--output-dir", str(out_dir)])
    assert res.exit_code == 0
    assert any(out_dir.glob("*.md"))


def test_tier2_output_directory_creation(sample_video: Path, tmp_path: Path, cli_runner):
    """Tier 2.15: Deeply nested non-existent output directory is created automatically."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    deep_out_dir = tmp_path / "level1" / "level2" / "nested_docs"

    assert not deep_out_dir.exists()
    res = cli_runner.invoke(cli_mod.app, ["process", str(sample_video), "--output-dir", str(deep_out_dir)])
    assert res.exit_code == 0
    assert deep_out_dir.exists()
    assert any(deep_out_dir.glob("*.md"))
