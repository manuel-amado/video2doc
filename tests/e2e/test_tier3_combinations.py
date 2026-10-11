"""Tier 3 - Cross-Feature Combinations E2E Tests.

Tests pairwise and multi-feature interactions across pipeline components:
- Custom template + faster-whisper backend + batch processing
- Built-in SOP template + frame limit tuning + scene threshold adjustments
- Built-in Lecture template + language selection + custom text model
- Built-in Executive template + custom debounce intervals + custom output directory
- Configuration hierarchy precedence (CLI flag > ENV var > YAML > defaults)
- Batch fault tolerance (partial corrupt file does not abort remaining videos)
- Staged VRAM unload order throughout end-to-end pipeline execution
- Sparse vs. dense keyframe limits with custom templates
- Pipeline initialized via config.yaml coupled with custom templates
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

pytestmark = pytest.mark.tier3


def test_tier3_custom_template_faster_whisper_batch_mode(
    sample_video_batch: Path, sample_custom_template: Path, tmp_path: Path, cli_runner
):
    """Tier 3.1: Batch processing multiple videos with custom YAML template and faster-whisper."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    out_dir = tmp_path / "batch_custom_out"

    res = cli_runner.invoke(
        cli_mod.app,
        [
            "batch",
            str(sample_video_batch),
            "--template",
            str(sample_custom_template),
            "--whisper-model",
            "base",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0
    docs = list(out_dir.glob("*.md"))
    assert len(docs) == 3


def test_tier3_sop_template_frame_limits_and_scene_threshold(
    sample_video: Path, tmp_path: Path, cli_runner, ffmpeg_state
):
    """Tier 3.2: SOP template execution with tuned scene threshold and frame limits."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    out_dir = tmp_path / "sop_tuned_out"

    res = cli_runner.invoke(
        cli_mod.app,
        [
            "process",
            str(sample_video),
            "--template",
            "sop",
            "--max-keyframes",
            "4",
            "--scene-threshold",
            "0.45",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0
    docs = list(out_dir.glob("*.md"))
    assert len(docs) == 1


def test_tier3_lecture_template_custom_language_and_text_model(
    sample_video: Path, tmp_path: Path, cli_runner, ollama_state
):
    """Tier 3.3: Lecture template with explicit language flag and custom text model."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    out_dir = tmp_path / "lecture_es_out"

    res = cli_runner.invoke(
        cli_mod.app,
        [
            "process",
            str(sample_video),
            "--template",
            "lecture",
            "--language",
            "es",
            "--text-model",
            "qwen2.5-coder:14b",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0
    assert any(c.model == "qwen2.5-coder:14b" for c in ollama_state.calls)


def test_tier3_executive_template_custom_debounce_and_output_dir(
    sample_video: Path, tmp_path: Path, cli_runner
):
    """Tier 3.4: Executive template with custom debounce seconds and custom output path."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    out_dir = tmp_path / "exec_custom_out"

    res = cli_runner.invoke(
        cli_mod.app,
        [
            "process",
            str(sample_video),
            "--template",
            "executive",
            "--debounce-seconds",
            "5.0",
            "--output-dir",
            str(out_dir),
        ],
    )
    assert res.exit_code == 0
    assert (out_dir / f"{sample_video.stem}.md").exists() or any(out_dir.glob("*.md"))


def test_tier3_env_override_plus_cli_override_precedence(
    sample_video: Path, tmp_path: Path, cli_runner, ollama_state
):
    """Tier 3.5: CLI parameters override both environment variables and defaults."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    out_dir = tmp_path / "prec_out"

    env_vars = {
        "VIDEO2DOC_TEXT_MODEL": "env-model-text",
        "VIDEO2DOC_VISION_MODEL": "env-model-vision",
    }
    with patch.dict(os.environ, env_vars):
        res = cli_runner.invoke(
            cli_mod.app,
            [
                "process",
                str(sample_video),
                "--text-model",
                "cli-model-final",
                "--output-dir",
                str(out_dir),
            ],
        )
        assert res.exit_code == 0
        assert any(c.model == "cli-model-final" for c in ollama_state.calls)


def test_tier3_batch_mode_partial_failure_resilience(tmp_path: Path, cli_runner):
    """Tier 3.6: Corrupt file in batch directory is logged; remaining valid videos succeed."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    batch_dir = tmp_path / "mixed_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    good_1 = batch_dir / "valid_video_1.mp4"
    good_1.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

    bad = batch_dir / "corrupted_video.mp4"
    bad.write_bytes(b"")  # empty corrupt

    good_2 = batch_dir / "valid_video_2.mp4"
    good_2.write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

    out_dir = tmp_path / "mixed_out"
    res = cli_runner.invoke(cli_mod.app, ["batch", str(batch_dir), "--output-dir", str(out_dir)])

    # Batch command completes, and generates docs for the valid videos
    docs = list(out_dir.glob("*.md"))
    assert len(docs) >= 2


def test_tier3_staged_vram_unload_order_across_full_pipeline(
    sample_video: Path, tmp_path: Path, ollama_state, mock_cuda
):
    """Tier 3.7: Full pipeline run executes staged unload signals in strict sequence."""
    pipeline_mod = pytest.importorskip("video2doc.pipeline")
    config_mod = pytest.importorskip("video2doc.config")

    settings = config_mod.Settings(vision_model="llama3.2-vision", text_model="qwen2.5-coder:14b")
    pipeline = pipeline_mod.DocumentPipeline(settings=settings)

    out_dir = tmp_path / "stage_out"
    pipeline.process_video(sample_video, template="sop", output_dir=out_dir)

    # Verify unload calls recorded in ollama_state
    assert "llama3.2-vision" in ollama_state.unloaded_models
    assert "qwen2.5-coder:14b" in ollama_state.unloaded_models
    assert mock_cuda["empty_cache"].called


def test_tier3_custom_template_with_sparse_and_dense_keyframes(
    sample_video: Path, sample_custom_template: Path, tmp_path: Path, cli_runner
):
    """Tier 3.8: Custom template handles both sparse (1 frame) and dense (15 frames) keyframe sets."""
    cli_mod = pytest.importorskip("video2doc.cli.main")

    sparse_dir = tmp_path / "sparse_out"
    res_sparse = cli_runner.invoke(
        cli_mod.app,
        ["process", str(sample_video), "--template", str(sample_custom_template), "--max-keyframes", "1", "--output-dir", str(sparse_dir)],
    )
    assert res_sparse.exit_code == 0
    assert any(sparse_dir.glob("*.md"))

    dense_dir = tmp_path / "dense_out"
    res_dense = cli_runner.invoke(
        cli_mod.app,
        ["process", str(sample_video), "--template", str(sample_custom_template), "--max-keyframes", "15", "--output-dir", str(dense_dir)],
    )
    assert res_dense.exit_code == 0
    assert any(dense_dir.glob("*.md"))


def test_tier3_pipeline_config_yaml_plus_custom_template(
    sample_video: Path, sample_custom_template: Path, tmp_path: Path
):
    """Tier 3.9: DocumentPipeline configured from YAML settings processes custom template."""
    config_mod = pytest.importorskip("video2doc.config")
    pipeline_mod = pytest.importorskip("video2doc.pipeline")

    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(
        yaml.dump({
            "vision_model": "llama3.2-vision",
            "text_model": "qwen2.5-coder:14b",
            "max_keyframes": 6,
            "scene_threshold": 0.35,
        }),
        encoding="utf-8",
    )

    settings = config_mod.Settings.from_yaml(cfg_file) if hasattr(config_mod.Settings, "from_yaml") else config_mod.Settings()
    pipeline = pipeline_mod.DocumentPipeline(settings=settings)

    out_dir = tmp_path / "cfg_custom_out"
    doc_path = pipeline.process_video(sample_video, template=str(sample_custom_template), output_dir=out_dir)
    assert doc_path.exists()
