"""Tier 4 - Real-World Application Scenarios E2E Tests.

Exercises complete, production-grade end-to-end workflows:
1. Software SOP / Tutorial Workflow (Technical procedures with UI screenshots and steps)
2. Academic Lecture & Research Notes Workflow (Theoretical notes with equations and summaries)
3. Executive Briefing Workflow (High-level decisions, KPI matrices, and action items)
4. Batch Course Curriculum Workflow (Multi-module educational video series)
5. Enterprise Custom Brand Standard Workflow (Custom organizational template with compliance checks)
"""

from pathlib import Path
import pytest

pytestmark = pytest.mark.tier4


def test_tier4_software_sop_workflow(sample_video: Path, tmp_path: Path, cli_runner, ollama_state, mock_cuda):
    """Tier 4.1: End-to-end Software SOP workflow converting screen recording into technical procedure manual."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    docs_dir = tmp_path / "sop_manuals"

    res = cli_runner.invoke(
        cli_mod.app,
        [
            "process",
            str(sample_video),
            "--template",
            "sop",
            "--max-keyframes",
            "6",
            "--scene-threshold",
            "0.3",
            "--output-dir",
            str(docs_dir),
        ],
    )
    assert res.exit_code == 0

    # 1. Output document exists
    doc_file = docs_dir / f"{sample_video.stem}.md"
    if not doc_file.exists():
        created_docs = list(docs_dir.glob("*.md"))
        assert len(created_docs) > 0
        doc_file = created_docs[0]

    content = doc_file.read_text(encoding="utf-8")

    # 2. Key sections present
    assert "# " in content
    assert "Overview" in content or "Instructions" in content or "Step" in content

    # 3. Model unload signals sent
    assert len(ollama_state.unloaded_models) >= 1
    assert mock_cuda["empty_cache"].called


def test_tier4_academic_lecture_workflow(sample_video: Path, tmp_path: Path, cli_runner, ollama_state):
    """Tier 4.2: Academic Lecture workflow converting lecture presentation into structured notes."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    notes_dir = tmp_path / "lecture_notes"

    res = cli_runner.invoke(
        cli_mod.app,
        [
            "process",
            str(sample_video),
            "--template",
            "lecture",
            "--whisper-model",
            "turbo",
            "--text-model",
            "qwen2.5-coder:14b",
            "--output-dir",
            str(notes_dir),
        ],
    )
    assert res.exit_code == 0

    created_docs = list(notes_dir.glob("*.md"))
    assert len(created_docs) > 0
    content = created_docs[0].read_text(encoding="utf-8")

    # Academic structure
    assert "#" in content
    assert any(c.model == "qwen2.5-coder:14b" for c in ollama_state.calls)


def test_tier4_executive_briefing_workflow(sample_video: Path, tmp_path: Path, cli_runner):
    """Tier 4.3: Executive Briefing workflow generating strategic decision and action item summaries."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    brief_dir = tmp_path / "exec_briefings"

    res = cli_runner.invoke(
        cli_mod.app,
        [
            "process",
            str(sample_video),
            "--template",
            "executive",
            "--max-keyframes",
            "4",
            "--output-dir",
            str(brief_dir),
        ],
    )
    assert res.exit_code == 0

    created_docs = list(brief_dir.glob("*.md"))
    assert len(created_docs) > 0
    content = created_docs[0].read_text(encoding="utf-8")
    assert len(content.strip()) > 0


def test_tier4_batch_curriculum_production_workflow(tmp_path: Path, cli_runner):
    """Tier 4.4: Batch workflow processing a multi-part video curriculum series."""
    cli_mod = pytest.importorskip("video2doc.cli.main")

    curriculum_dir = tmp_path / "course_videos"
    curriculum_dir.mkdir(parents=True, exist_ok=True)
    modules = ["01_introduction.mp4", "02_core_concepts.mp4", "03_lab_exercise.mp4", "04_conclusion.mp4"]
    for m in modules:
        (curriculum_dir / m).write_bytes(b"\x00\x00\x00 ftypisom\x00\x00\x02\x00isomiso2avc1mp41")

    out_dir = tmp_path / "course_docs"
    res = cli_runner.invoke(
        cli_mod.app,
        ["batch", str(curriculum_dir), "--template", "lecture", "--output-dir", str(out_dir)],
    )
    assert res.exit_code == 0

    generated_docs = list(out_dir.glob("*.md"))
    assert len(generated_docs) == 4


def test_tier4_enterprise_custom_brand_workflow(sample_video: Path, tmp_path: Path, cli_runner):
    """Tier 4.5: Enterprise workflow with custom brand template and compliance verification."""
    cli_mod = pytest.importorskip("video2doc.cli.main")

    brand_tmpl = tmp_path / "enterprise_sop.yaml"
    brand_tmpl.write_text(
        """name: enterprise_sop
description: Corporate Enterprise Standard Operating Procedure
vision_prompt: >
  Inspect the technical console and configuration screenshot.
  Verify compliance with ISO security standards and identify all open network ports.
synthesis_prompt: >
  # Corporate Operational Manual: {{ title }}
  
  ## Executive Overview
  {{ transcription_text }}
  
  ## Visual Compliance Findings
  {{ visual_summary }}
  
  ## QA & Security Sign-off Checklist
  - [ ] Security boundaries validated
  - [ ] Role-based access configured
  - [ ] Automated regression tests passed
sections:
  - Executive Overview
  - Visual Compliance Findings
  - QA & Security Sign-off Checklist
""",
        encoding="utf-8",
    )

    out_dir = tmp_path / "enterprise_docs"
    res = cli_runner.invoke(
        cli_mod.app,
        ["process", str(sample_video), "--template", str(brand_tmpl), "--output-dir", str(out_dir)],
    )
    assert res.exit_code == 0

    created_docs = list(out_dir.glob("*.md"))
    assert len(created_docs) > 0
    content = created_docs[0].read_text(encoding="utf-8")
    assert "Corporate Operational Manual" in content or "Security" in content or "Sign-off" in content
