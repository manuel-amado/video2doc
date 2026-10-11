"""Unit tests for the Video2Doc Typer CLI application."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from video2doc.cli.main import app, check_environment, run_diagnostics
from video2doc.config import Settings
from video2doc.exceptions import (
    MediaCorruptError,
    MissingDependencyError,
    ModelNotFoundError,
    OllamaConnectionError,
    TemplateNotFoundError,
    Video2DocError,
)

runner = CliRunner()


# ==============================================================================
# 1. Help Commands Verification
# ==============================================================================

def test_cli_help_root() -> None:
    """Root CLI --help displays description and all subcommands with exit code 0."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "video2doc" in result.output.lower() or "multimodal" in result.output.lower()
    for subcmd in ["process", "batch", "templates", "config"]:
        assert subcmd in result.output


def test_cli_process_help() -> None:
    """CLI process --help displays typed options and arguments with exit code 0."""
    result = runner.invoke(app, ["process", "--help"])
    assert result.exit_code == 0
    assert "--template" in result.output
    assert "--output" in result.output or "--output-dir" in result.output
    assert "--model-vision" in result.output
    assert "--model-text" in result.output
    assert "--whisper-model" in result.output
    assert "--scene-threshold" in result.output


def test_cli_batch_help() -> None:
    """CLI batch --help displays options for batch directory processing."""
    result = runner.invoke(app, ["batch", "--help"])
    assert result.exit_code == 0
    assert "--template" in result.output
    assert "--output" in result.output or "--output-dir" in result.output
    assert "--input-dir" in result.output


def test_cli_templates_help() -> None:
    """CLI templates --help displays subcommands (list, show, init)."""
    result = runner.invoke(app, ["templates", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "show" in result.output
    assert "init" in result.output


def test_cli_config_help() -> None:
    """CLI config --help displays subcommands (show, check)."""
    result = runner.invoke(app, ["config", "--help"])
    assert result.exit_code == 0
    assert "show" in result.output
    assert "check" in result.output


# ==============================================================================
# 2. Process Command Execution & Options
# ==============================================================================

def test_cli_process_happy_path(tmp_path: Path) -> None:
    """Process command executes pipeline and outputs generated markdown path."""
    video_file = tmp_path / "sample.mp4"
    video_file.write_bytes(b"dummy video content")
    out_dir = tmp_path / "output"
    expected_doc = out_dir / "sample.md"

    with patch("video2doc.cli.main.DocumentPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.process_video.return_value = expected_doc
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(
            app,
            [
                "process",
                str(video_file),
                "--template",
                "sop",
                "--output-dir",
                str(out_dir),
            ],
        )

        assert result.exit_code == 0
        assert "sample.md" in result.output
        mock_pipeline.process_video.assert_called_once()


def test_cli_process_with_all_flags(tmp_path: Path) -> None:
    """Process command parses and passes all custom flags into pipeline settings."""
    video_file = tmp_path / "lecture_demo.mp4"
    video_file.write_bytes(b"dummy video content")
    out_dir = tmp_path / "output_custom"
    expected_doc = out_dir / "lecture_demo.md"

    with patch("video2doc.cli.main.DocumentPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.process_video.return_value = expected_doc
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(
            app,
            [
                "process",
                "-i",
                str(video_file),
                "-t",
                "lecture",
                "-o",
                str(out_dir),
                "--model-vision",
                "custom-vision-llm",
                "--model-text",
                "custom-text-llm",
                "--whisper-model",
                "small",
                "--whisper-device",
                "cpu",
                "-l",
                "en",
                "--scene-threshold",
                "0.45",
                "--max-frames",
                "15",
                "--debounce-seconds",
                "4.5",
                "--keep-frames",
                "--embed-images",
                "--ollama-url",
                "http://localhost:11434",
            ],
        )

        assert result.exit_code == 0
        # Check settings passed to DocumentPipeline constructor
        init_kwargs = mock_pipeline_cls.call_args[1]
        settings: Settings = init_kwargs["settings"]
        assert settings.vision_model == "custom-vision-llm"
        assert settings.text_model == "custom-text-llm"
        assert settings.whisper_model == "small"
        assert settings.whisper_device == "cpu"
        assert settings.whisper_language == "en"
        assert settings.scene_threshold == 0.45
        assert settings.max_keyframes == 15
        assert settings.debounce_seconds == 4.5
        assert settings.keep_frames is True
        assert settings.embed_images is True


def test_cli_process_missing_input_argument() -> None:
    """Process command without input video exits with non-zero code and message."""
    result = runner.invoke(app, ["process"])
    assert result.exit_code != 0
    assert "No input video" in result.output or "Missing" in result.output


def test_cli_process_nonexistent_file_handling(tmp_path: Path) -> None:
    """Process command with non-existent video catches MediaCorruptError and exits cleanly."""
    missing = tmp_path / "does_not_exist.mp4"
    result = runner.invoke(app, ["process", str(missing)])
    assert result.exit_code != 0
    assert "MediaCorruptError" in result.output or "does not exist" in result.output


def test_cli_process_handles_ollama_connection_error(tmp_path: Path) -> None:
    """Process command catches OllamaConnectionError gracefully without crash traceback."""
    video = tmp_path / "test.mp4"
    video.write_bytes(b"content")

    with patch("video2doc.cli.main.DocumentPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.process_video.side_effect = OllamaConnectionError(
            "http://localhost:11434", reason="Connection refused"
        )
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(app, ["process", str(video)])
        assert result.exit_code == 1
        assert "OllamaConnectionError" in result.output
        assert "Connection refused" in result.output


def test_cli_process_handles_model_not_found_error(tmp_path: Path) -> None:
    """Process command catches ModelNotFoundError gracefully."""
    video = tmp_path / "test.mp4"
    video.write_bytes(b"content")

    with patch("video2doc.cli.main.DocumentPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.process_video.side_effect = ModelNotFoundError(
            "llama3.2-vision", available_models=["llava", "qwen"]
        )
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(app, ["process", str(video)])
        assert result.exit_code == 1
        assert "ModelNotFoundError" in result.output


def test_cli_process_handles_missing_dependency_error(tmp_path: Path) -> None:
    """Process command catches MissingDependencyError gracefully."""
    video = tmp_path / "test.mp4"
    video.write_bytes(b"content")

    with patch("video2doc.cli.main.DocumentPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.process_video.side_effect = MissingDependencyError(
            "ffmpeg", remediation="Install ffmpeg via winget."
        )
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(app, ["process", str(video)])
        assert result.exit_code == 1
        assert "MissingDependencyError" in result.output
        assert "Install ffmpeg" in result.output


def test_cli_process_handles_template_not_found_error(tmp_path: Path) -> None:
    """Process command catches TemplateNotFoundError gracefully."""
    video = tmp_path / "test.mp4"
    video.write_bytes(b"content")

    with patch("video2doc.cli.main.DocumentPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.process_video.side_effect = TemplateNotFoundError("unknown_template")
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(app, ["process", str(video), "-t", "unknown_template"])
        assert result.exit_code == 1
        assert "TemplateNotFoundError" in result.output


# ==============================================================================
# 3. Batch Command Execution
# ==============================================================================

def test_cli_batch_happy_path(tmp_path: Path) -> None:
    """Batch command processes folder and displays summary table with exit code 0."""
    batch_dir = tmp_path / "videos"
    batch_dir.mkdir()
    out_dir = tmp_path / "docs"

    with patch("video2doc.cli.main.DocumentPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.process_batch.return_value = [
            out_dir / "vid1.md",
            out_dir / "vid2.md",
        ]
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(
            app,
            ["batch", str(batch_dir), "--output-dir", str(out_dir)],
        )

        assert result.exit_code == 0
        assert "vid1.md" in result.output
        assert "vid2.md" in result.output
        assert "2 document(s) generated successfully" in result.output


def test_cli_batch_missing_input_directory() -> None:
    """Batch command without input directory exits with error code."""
    result = runner.invoke(app, ["batch"])
    assert result.exit_code != 0
    assert "No input directory" in result.output or "Missing" in result.output


def test_cli_batch_error_handling(tmp_path: Path) -> None:
    """Batch command handles Video2DocError and reports failure cleanly."""
    batch_dir = tmp_path / "videos"
    batch_dir.mkdir()

    with patch("video2doc.cli.main.DocumentPipeline") as mock_pipeline_cls:
        mock_pipeline = MagicMock()
        mock_pipeline.process_batch.side_effect = Video2DocError("Batch processing failure")
        mock_pipeline_cls.return_value = mock_pipeline

        result = runner.invoke(app, ["batch", str(batch_dir)])
        assert result.exit_code == 1
        assert "Batch processing failure" in result.output


# ==============================================================================
# 4. Templates Commands
# ==============================================================================

def test_cli_templates_list() -> None:
    """Templates list displays built-in templates (sop, lecture, executive)."""
    result = runner.invoke(app, ["templates", "list"])
    assert result.exit_code == 0
    assert "sop" in result.output
    assert "lecture" in result.output
    assert "executive" in result.output


def test_cli_templates_show_valid() -> None:
    """Templates show displays details of built-in sop template."""
    result = runner.invoke(app, ["templates", "show", "sop"])
    assert result.exit_code == 0
    assert "sop" in result.output
    assert "Vision Prompt" in result.output
    assert "Synthesis Prompt" in result.output


def test_cli_templates_show_invalid() -> None:
    """Templates show with non-existent template exits with error."""
    result = runner.invoke(app, ["templates", "show", "non_existent_xyz"])
    assert result.exit_code != 0
    assert "TemplateNotFoundError" in result.output


def test_cli_templates_init(tmp_path: Path) -> None:
    """Templates init scaffolds starter template YAML file."""
    dest = tmp_path / "custom.yaml"
    result = runner.invoke(app, ["templates", "init", str(dest)])
    assert result.exit_code == 0
    assert dest.exists()
    content = dest.read_text(encoding="utf-8")
    assert "my_custom_template" in content
    assert "vision_prompt" in content
    assert "synthesis_prompt" in content


# ==============================================================================
# 5. Config Commands & Diagnostics
# ==============================================================================

def test_cli_config_show() -> None:
    """Config show displays active settings keys and values."""
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "ollama_url" in result.output
    assert "vision_model" in result.output
    assert "11434" in result.output


def test_cli_config_check_all_ok() -> None:
    """Config check displays passing status when all dependencies are available."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "models": [{"name": "llama3.2-vision"}, {"name": "qwen2.5-coder:14b"}]
    }

    with patch("shutil.which", return_value="/usr/bin/tool"), patch(
        "requests.get", return_value=mock_response
    ):
        result = runner.invoke(app, ["config", "check"])
        assert result.exit_code == 0
        assert "OK" in result.output or "AVAILABLE" in result.output


def test_cli_config_check_missing_dependencies() -> None:
    """Config check displays diagnostic warnings when tools or Ollama are missing."""
    with patch("shutil.which", return_value=None), patch(
        "requests.get", side_effect=Exception("Connection refused")
    ):
        result = runner.invoke(app, ["config", "check"])
        assert result.exit_code == 0
        assert "MISSING" in result.output or "UNREACHABLE" in result.output


def test_check_environment_helper() -> None:
    """Direct invocation of check_environment helper returns structured diagnostic map."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "models": [{"name": "llama3.2-vision:latest"}, {"name": "qwen2.5-coder:14b"}]
    }

    with patch("shutil.which", return_value="/fake/path"), patch(
        "requests.get", return_value=mock_response
    ):
        diag = check_environment()
        assert diag["ffmpeg"] is True
        assert diag["ffprobe"] is True
        assert diag["ollama"] is True
        assert diag["vision_model"] is True
        assert diag["text_model"] is True
        assert diag["all_ok"] is True

    # Test alias
    assert run_diagnostics is check_environment
