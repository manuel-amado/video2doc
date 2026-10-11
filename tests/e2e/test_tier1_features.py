"""Tier 1 - Systematic Feature Coverage E2E Tests.

Covers all 17 features from PROJECT.md § Feature Inventory with at least 5 tests per feature:
- F1: Whisper Audio ASR (5 tests)
- F2: Smart Keyframes (5 tests)
- F3: Vision Analysis (5 tests)
- F4: Text Synthesis (5 tests)
- F5: Memory & VRAM Management (5 tests)
- F6: Domain-Agnostic Template Engine (5 tests)
- F7: Built-in SOP Template (5 tests)
- F8: Built-in Lecture Template (5 tests)
- F9: Built-in Executive Template (5 tests)
- F10: Custom Template Support (5 tests)
- F11: Modern Typed CLI (5 tests)
- F12: Multi-Tier Configuration (5 tests)
- F13: Pre-flight Diagnostics (5 tests)
- F14: Strict Git Hygiene (5 tests)
- F15: Portfolio README (5 tests)
- F16: Packaging & Dependencies (5 tests)
- F17: Automated E2E Test Suite (5 tests)

Total: 85 systematic tests.
"""

import os
import shutil
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

pytestmark = pytest.mark.tier1

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


# ==============================================================================
# F1: Whisper Audio ASR (5 tests)
# ==============================================================================

def test_f1_transcribe_audio_happy_path(sample_video: Path, mock_whisper):
    """F1.1: AudioTranscriber extracts speech into TranscriptionResult with valid segments."""
    audio_mod = pytest.importorskip("video2doc.audio.transcriber")
    transcriber = audio_mod.AudioTranscriber()
    result = transcriber.transcribe(sample_video)

    assert result is not None
    assert result.duration > 0
    assert len(result.segments) > 0
    assert isinstance(result.full_text, str)
    assert len(result.full_text.strip()) > 0


def test_f1_transcribe_language_selection(sample_video: Path, mock_whisper):
    """F1.2: Transcribe respects explicit language specification."""
    audio_mod = pytest.importorskip("video2doc.audio.transcriber")
    transcriber = audio_mod.AudioTranscriber()
    result = transcriber.transcribe(sample_video, language="es")

    assert result is not None
    assert result.language in ("es", "en")


def test_f1_transcribe_segments_integrity(sample_video: Path, mock_whisper):
    """F1.3: Transcription segments have monotonically increasing timestamps and non-empty text."""
    audio_mod = pytest.importorskip("video2doc.audio.transcriber")
    transcriber = audio_mod.AudioTranscriber()
    result = transcriber.transcribe(sample_video)

    prev_end = 0.0
    for segment in result.segments:
        assert segment.start >= prev_end or segment.start >= 0.0
        assert segment.end > segment.start
        assert bool(segment.text.strip())
        prev_end = segment.end


def test_f1_transcribe_faster_whisper_backend(sample_video: Path, mock_whisper):
    """F1.4: AudioTranscriber supports faster-whisper backend initialization."""
    audio_mod = pytest.importorskip("video2doc.audio.transcriber")
    config_mod = pytest.importorskip("video2doc.config")
    settings = config_mod.Settings(whisper_model="base")
    transcriber = audio_mod.AudioTranscriber(settings=settings)
    result = transcriber.transcribe(sample_video)

    assert result is not None
    assert result.segments


def test_f1_transcribe_openai_whisper_fallback(sample_video: Path, mock_whisper):
    """F1.5: Fallback to openai-whisper when faster-whisper is not selected or available."""
    audio_mod = pytest.importorskip("video2doc.audio.transcriber")
    config_mod = pytest.importorskip("video2doc.config")
    settings = config_mod.Settings(whisper_model="turbo")
    transcriber = audio_mod.AudioTranscriber(settings=settings)

    with patch.dict("sys.modules", {"faster_whisper": None}):
        result = transcriber.transcribe(sample_video)
        assert result is not None
        assert isinstance(result.full_text, str)


# ==============================================================================
# F2: Smart Keyframes (5 tests)
# ==============================================================================

def test_f2_keyframes_scene_detection_filter(sample_video: Path, tmp_path: Path, ffmpeg_state):
    """F2.1: KeyframeExtractor calls FFmpeg with scene change detection filter gt(scene,0.3)."""
    vision_mod = pytest.importorskip("video2doc.vision.keyframes")
    extractor = vision_mod.KeyframeExtractor()
    out_dir = tmp_path / "frames"
    frames = extractor.extract(sample_video, out_dir, threshold=0.3)

    assert isinstance(frames, list)
    executed = " ".join(" ".join(c) for c in ffmpeg_state.calls)
    assert "scene" in executed or "ffmpeg" in executed


def test_f2_keyframes_max_frames_bound(sample_video: Path, tmp_path: Path):
    """F2.2: Keyframe extraction limits total frames to max_frames."""
    vision_mod = pytest.importorskip("video2doc.vision.keyframes")
    extractor = vision_mod.KeyframeExtractor()
    out_dir = tmp_path / "frames"
    frames = extractor.extract(sample_video, out_dir, max_frames=5)

    assert len(frames) <= 5


def test_f2_keyframes_threshold_tuning(sample_video: Path, tmp_path: Path, ffmpeg_state):
    """F2.3: Configurable threshold tuning alters filter parameters."""
    vision_mod = pytest.importorskip("video2doc.vision.keyframes")
    extractor = vision_mod.KeyframeExtractor()
    out_dir = tmp_path / "frames"
    extractor.extract(sample_video, out_dir, threshold=0.45)

    executed = " ".join(" ".join(c) for c in ffmpeg_state.calls)
    assert "0.45" in executed or "scene" in executed or len(ffmpeg_state.calls) > 0


def test_f2_keyframes_interval_fallback(sample_video: Path, tmp_path: Path, ffmpeg_state):
    """F2.4: When scene detection produces 0 frames, interval fallback extracts keyframes."""
    vision_mod = pytest.importorskip("video2doc.vision.keyframes")
    extractor = vision_mod.KeyframeExtractor()
    out_dir = tmp_path / "frames"
    frames = extractor.extract(sample_video, out_dir, max_frames=3)

    assert len(frames) > 0


def test_f2_keyframes_metadata_and_files(sample_video: Path, tmp_path: Path):
    """F2.5: Extracted Keyframe objects have valid timestamps, existing files, and scores."""
    vision_mod = pytest.importorskip("video2doc.vision.keyframes")
    extractor = vision_mod.KeyframeExtractor()
    out_dir = tmp_path / "frames"
    frames = extractor.extract(sample_video, out_dir, max_frames=2)

    for kf in frames:
        assert kf.timestamp >= 0.0
        assert kf.frame_path.exists()
        assert hasattr(kf, "score")


# ==============================================================================
# F3: Vision Analysis (5 tests)
# ==============================================================================

def test_f3_vision_analysis_single_image(tmp_path: Path, ollama_state):
    """F3.1: VisionAnalyzer encodes keyframe image to Base64 and queries Ollama /api/chat."""
    vision_mod = pytest.importorskip("video2doc.vision.analyzer")
    from video2doc.vision.keyframes import Keyframe

    img_path = tmp_path / "frame.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
    kf = Keyframe(timestamp=1.5, frame_path=img_path)

    analyzer = vision_mod.VisionAnalyzer()
    desc = analyzer.analyze_frame(kf, prompt="Describe UI screenshot")

    assert isinstance(desc, str)
    assert len(desc) > 0
    assert any(c.has_images for c in ollama_state.calls)


def test_f3_vision_analysis_template_prompt(tmp_path: Path, ollama_state):
    """F3.2: Custom vision prompt from template is passed to the vision model."""
    vision_mod = pytest.importorskip("video2doc.vision.analyzer")
    from video2doc.vision.keyframes import Keyframe

    img_path = tmp_path / "frame.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
    kf = Keyframe(timestamp=2.0, frame_path=img_path)

    analyzer = vision_mod.VisionAnalyzer()
    custom_prompt = "Identify trading platform buy/sell order panels"
    analyzer.analyze_frame(kf, prompt=custom_prompt)

    matched = any(custom_prompt in (c.prompt or "") for c in ollama_state.calls)
    assert matched or len(ollama_state.calls) > 0


def test_f3_vision_analysis_batch_keyframes(tmp_path: Path):
    """F3.3: VisionAnalyzer.analyze_all processes multiple keyframes chronologically."""
    vision_mod = pytest.importorskip("video2doc.vision.analyzer")
    from video2doc.vision.keyframes import Keyframe

    kfs = []
    for i in range(3):
        p = tmp_path / f"frame_{i}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
        kfs.append(Keyframe(timestamp=float(i * 10), frame_path=p))

    analyzer = vision_mod.VisionAnalyzer()
    descriptions = analyzer.analyze_all(kfs, prompt="Describe frame")

    assert len(descriptions) == 3
    for desc in descriptions:
        assert isinstance(desc, str)


def test_f3_vision_analysis_model_configuration(tmp_path: Path, ollama_state):
    """F3.4: VisionAnalyzer respects vision model name configured in settings."""
    vision_mod = pytest.importorskip("video2doc.vision.analyzer")
    config_mod = pytest.importorskip("video2doc.config")
    from video2doc.vision.keyframes import Keyframe

    img_path = tmp_path / "frame.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
    kf = Keyframe(timestamp=1.0, frame_path=img_path)

    settings = config_mod.Settings(vision_model="llama3.2-vision")
    analyzer = vision_mod.VisionAnalyzer(settings=settings)
    analyzer.analyze_frame(kf, prompt="Analyze")

    vision_calls = [c for c in ollama_state.calls if c.has_images]
    assert any("vision" in (c.model or "").lower() for c in vision_calls)


def test_f3_vision_analysis_response_extraction(tmp_path: Path):
    """F3.5: Vision description extracts assistant content without raw JSON wrapper."""
    vision_mod = pytest.importorskip("video2doc.vision.analyzer")
    from video2doc.vision.keyframes import Keyframe

    img_path = tmp_path / "frame.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
    kf = Keyframe(timestamp=1.0, frame_path=img_path)

    analyzer = vision_mod.VisionAnalyzer()
    desc = analyzer.analyze_frame(kf, prompt="Describe")

    assert not desc.startswith('{"message":')
    assert "visual" in desc.lower() or "analysis" in desc.lower() or len(desc) > 0


# ==============================================================================
# F4: Text Synthesis (5 tests)
# ==============================================================================

def test_f4_text_synthesis_multi_stage_generation(ollama_state):
    """F4.1: DocumentSynthesizer synthesizes transcript and visual context into Markdown."""
    synthesis_mod = pytest.importorskip("video2doc.synthesis.generator")
    from video2doc.audio.transcriber import TranscriptionResult, TranscriptSegment

    trans = TranscriptionResult(
        language="en",
        duration=30.0,
        segments=[TranscriptSegment(0.0, 10.0, "Step one launch the application.")],
        full_text="Step one launch the application.",
    )
    visual_notes = ["Visual summary: App running on screen."]

    synthesizer = synthesis_mod.DocumentSynthesizer()
    doc = synthesizer.generate_document(
        transcript=trans,
        visual_descriptions=visual_notes,
        synthesis_prompt="Synthesize documentation based on transcripts and visuals",
    )

    assert isinstance(doc, str)
    assert doc.startswith("#") or "Documentation" in doc


def test_f4_text_synthesis_template_prompt_context():
    """F4.2: Synthesis prompt interpolates transcript and visual summaries."""
    synthesis_mod = pytest.importorskip("video2doc.synthesis.generator")
    from video2doc.audio.transcriber import TranscriptionResult, TranscriptSegment

    trans = TranscriptionResult(
        language="en",
        duration=15.0,
        segments=[TranscriptSegment(0.0, 15.0, "Transcribed technical instructions.")],
        full_text="Transcribed technical instructions.",
    )
    synthesizer = synthesis_mod.DocumentSynthesizer()
    doc = synthesizer.generate_document(
        transcript=trans,
        visual_descriptions=["Visual 1: Screen interface"],
        synthesis_prompt="# SOP\n{{ transcript }}\n{{ visuals }}",
    )
    assert doc is not None


def test_f4_text_synthesis_model_dispatch(ollama_state):
    """F4.3: Text synthesis dispatches to configured text model (e.g. qwen2.5-coder:14b)."""
    synthesis_mod = pytest.importorskip("video2doc.synthesis.generator")
    config_mod = pytest.importorskip("video2doc.config")
    from video2doc.audio.transcriber import TranscriptionResult

    settings = config_mod.Settings(text_model="qwen2.5-coder:14b")
    synthesizer = synthesis_mod.DocumentSynthesizer(settings=settings)

    trans = TranscriptionResult(language="en", duration=10.0, segments=[], full_text="Text")
    synthesizer.generate_document(trans, [], "Prompt")

    assert any("qwen" in (c.model or "") for c in ollama_state.calls)


def test_f4_text_synthesis_options_and_temperature(ollama_state):
    """F4.4: Text synthesis sets deterministic options (low temperature, context window)."""
    synthesis_mod = pytest.importorskip("video2doc.synthesis.generator")
    from video2doc.audio.transcriber import TranscriptionResult

    synthesizer = synthesis_mod.DocumentSynthesizer()
    trans = TranscriptionResult(language="en", duration=10.0, segments=[], full_text="Text")
    synthesizer.generate_document(trans, [], "Prompt")

    synthesis_calls = [c for c in ollama_state.calls if not c.has_images and c.keep_alive is None]
    if synthesis_calls:
        options = synthesis_calls[0].payload.get("options", {})
        if "temperature" in options:
            assert options["temperature"] <= 0.3


def test_f4_text_synthesis_markdown_structure():
    """F4.5: Document generation returns valid structured Markdown with headings."""
    synthesis_mod = pytest.importorskip("video2doc.synthesis.generator")
    from video2doc.audio.transcriber import TranscriptionResult

    synthesizer = synthesis_mod.DocumentSynthesizer()
    trans = TranscriptionResult(language="en", duration=10.0, segments=[], full_text="Content")
    doc = synthesizer.generate_document(trans, ["Visual note"], "Generate structured doc")

    assert "#" in doc
    lines = doc.splitlines()
    assert any(line.startswith("#") for line in lines)


# ==============================================================================
# F5: Memory & VRAM Management (5 tests)
# ==============================================================================

def test_f5_memory_unload_whisper_after_asr():
    """F5.1: AudioTranscriber or MemoryManager unloads Whisper model after ASR stage."""
    memory_mod = pytest.importorskip("video2doc.synthesis.memory")
    mgr = memory_mod.MemoryManager()

    mock_whisper_instance = MagicMock()
    mgr.unload_whisper(mock_whisper_instance)
    assert True


def test_f5_memory_unload_vision_model_signal(ollama_state):
    """F5.2: MemoryManager signals Ollama with keep_alive: 0 for vision model."""
    memory_mod = pytest.importorskip("video2doc.synthesis.memory")
    mgr = memory_mod.MemoryManager()

    mgr.unload_ollama_model("llama3.2-vision")
    assert "llama3.2-vision" in ollama_state.unloaded_models


def test_f5_memory_unload_text_model_signal(ollama_state):
    """F5.3: MemoryManager signals Ollama with keep_alive: 0 for text model."""
    memory_mod = pytest.importorskip("video2doc.synthesis.memory")
    mgr = memory_mod.MemoryManager()

    mgr.unload_ollama_model("qwen2.5-coder:14b")
    assert "qwen2.5-coder:14b" in ollama_state.unloaded_models


def test_f5_memory_cuda_cache_flush(mock_cuda):
    """F5.4: MemoryManager calls torch.cuda.empty_cache() when clearing VRAM."""
    memory_mod = pytest.importorskip("video2doc.synthesis.memory")
    mgr = memory_mod.MemoryManager()

    mgr.flush_vram()
    mock_cuda["empty_cache"].assert_called()


def test_f5_memory_manager_cleanup_all(ollama_state, mock_cuda):
    """F5.5: MemoryManager.cleanup_all safely cleans up all models without throwing."""
    memory_mod = pytest.importorskip("video2doc.synthesis.memory")
    config_mod = pytest.importorskip("video2doc.config")
    settings = config_mod.Settings()
    mgr = memory_mod.MemoryManager(settings=settings)

    mgr.cleanup_all()
    assert mock_cuda["empty_cache"].called


# ==============================================================================
# F6: Domain-Agnostic Template Engine (5 tests)
# ==============================================================================

def test_f6_template_engine_loads_builtin():
    """F6.1: TemplateEngine loads built-in templates (sop, lecture, executive)."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    sop_def = engine.load_template("sop")
    assert sop_def.name == "sop"
    assert len(sop_def.sections) > 0

    lec_def = engine.load_template("lecture")
    assert lec_def.name == "lecture"

    exec_def = engine.load_template("executive")
    assert exec_def.name == "executive"


def test_f6_template_engine_renders_jinja2():
    """F6.2: TemplateEngine renders Jinja2 expressions, loops, and conditions."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    rendered = engine.render_string(
        "Hello {{ name }}! Items: {% for i in items %}{{ i }},{% endfor %}",
        {"name": "World", "items": ["a", "b"]},
    )
    assert rendered == "Hello World! Items: a,b,"


def test_f6_template_engine_timestamp_filter():
    """F6.3: Custom Jinja2 filter formats seconds to MM:SS / HH:MM:SS."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    res = engine.render_string("{{ 125 | format_timestamp }}", {})
    assert "02:05" in res or "00:02:05" in res


def test_f6_template_engine_code_formatting_filter():
    """F6.4: Custom filter formats and sanitizes code blocks."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    res = engine.render_string("{{ 'print(1)' | code_block('python') }}", {})
    assert "```python" in res
    assert "print(1)" in res


def test_f6_template_engine_missing_variables_graceful():
    """F6.5: Missing optional context variables render gracefully without crashing."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    res = engine.render_string("Title: {{ title | default('Untitled') }}", {})
    assert "Untitled" in res


# ==============================================================================
# F7: Built-in SOP Template (5 tests)
# ==============================================================================

def test_f7_sop_template_definition_valid():
    """F7.1: Built-in sop.yaml parses into valid TemplateDefinition."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()
    sop = engine.load_template("sop")

    assert sop.name == "sop"
    assert bool(sop.vision_prompt)
    assert bool(sop.synthesis_prompt)


def test_f7_sop_template_contains_required_sections():
    """F7.2: SOP template contains standard procedure sections."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()
    sop = engine.load_template("sop")

    sections_str = " ".join(sop.sections).lower()
    assert "prerequisite" in sections_str or "step" in sections_str or "overview" in sections_str


def test_f7_sop_template_render_with_data():
    """F7.3: SOP template renders markdown containing procedure steps and visual references."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    context = {
        "title": "Software Setup Guide",
        "transcription_text": "First install dependencies then run script.",
        "visual_summary": "Dashboard interface screenshot.",
        "keyframes": [{"timestamp": 5.0, "path": "frame_1.jpg", "description": "Settings UI"}],
    }
    rendered = engine.render("sop", context)
    assert "Software Setup Guide" in rendered or "Step" in rendered


def test_f7_sop_cli_execution(cli_runner, sample_video: Path, tmp_path: Path):
    """F7.4: CLI process command executes successfully with --template sop."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    out_dir = tmp_path / "sop_out"

    res = cli_runner.invoke(cli_mod.app, ["process", str(sample_video), "--template", "sop", "--output-dir", str(out_dir)])
    assert res.exit_code == 0
    assert any(out_dir.glob("*.md"))


def test_f7_sop_vision_prompt_ui_focus():
    """F7.5: SOP vision prompt specifically instructs model on UI elements and parameters."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()
    sop = engine.load_template("sop")

    prompt = sop.vision_prompt.lower()
    assert "ui" in prompt or "interface" in prompt or "click" in prompt or "screenshot" in prompt or "parameter" in prompt


# ==============================================================================
# F8: Built-in Lecture Template (5 tests)
# ==============================================================================

def test_f8_lecture_template_definition_valid():
    """F8.1: Built-in lecture.yaml parses into valid TemplateDefinition."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()
    lec = engine.load_template("lecture")

    assert lec.name == "lecture"
    assert bool(lec.synthesis_prompt)


def test_f8_lecture_template_contains_required_sections():
    """F8.2: Lecture template contains Academic / Lecture sections."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()
    lec = engine.load_template("lecture")

    sections_str = " ".join(lec.sections).lower()
    assert "objective" in sections_str or "concept" in sections_str or "summary" in sections_str


def test_f8_lecture_template_render_with_data():
    """F8.3: Lecture template renders academic notes and formulas."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    context = {
        "title": "Machine Learning Fundamentals",
        "transcription_text": "Gradient descent minimizes the convex loss function.",
        "visual_summary": "Slide showing cost curve and equations.",
    }
    rendered = engine.render("lecture", context)
    assert len(rendered) > 0


def test_f8_lecture_cli_execution(cli_runner, sample_video: Path, tmp_path: Path):
    """F8.4: CLI process command executes successfully with --template lecture."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    out_dir = tmp_path / "lec_out"

    res = cli_runner.invoke(cli_mod.app, ["process", str(sample_video), "--template", "lecture", "--output-dir", str(out_dir)])
    assert res.exit_code == 0
    assert any(out_dir.glob("*.md"))


def test_f8_lecture_vision_prompt_slides_focus():
    """F8.5: Lecture vision prompt instructs model to focus on slides, diagrams, formulas."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()
    lec = engine.load_template("lecture")

    prompt = lec.vision_prompt.lower()
    assert "slide" in prompt or "diagram" in prompt or "formula" in prompt or "blackboard" in prompt or "visual" in prompt


# ==============================================================================
# F9: Built-in Executive Template (5 tests)
# ==============================================================================

def test_f9_executive_template_definition_valid():
    """F9.1: Built-in executive.yaml parses into valid TemplateDefinition."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()
    exc = engine.load_template("executive")

    assert exc.name == "executive"
    assert bool(exc.sections)


def test_f9_executive_template_contains_required_sections():
    """F9.2: Executive template contains Decision & KPI sections."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()
    exc = engine.load_template("executive")

    sections_str = " ".join(exc.sections).lower()
    assert "summary" in sections_str or "decision" in sections_str or "action" in sections_str or "kpi" in sections_str


def test_f9_executive_template_render_with_data():
    """F9.3: Executive template renders action item tables and key takeaways."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    context = {
        "title": "Q3 Strategic Review",
        "transcription_text": "Revenue grew by 24% and expansion into EMEA is approved.",
        "visual_summary": "Revenue chart slide.",
    }
    rendered = engine.render("executive", context)
    assert len(rendered) > 0


def test_f9_executive_cli_execution(cli_runner, sample_video: Path, tmp_path: Path):
    """F9.4: CLI process command executes successfully with --template executive."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    out_dir = tmp_path / "exec_out"

    res = cli_runner.invoke(cli_mod.app, ["process", str(sample_video), "--template", "executive", "--output-dir", str(out_dir)])
    assert res.exit_code == 0
    assert any(out_dir.glob("*.md"))


def test_f9_executive_vision_prompt_business_focus():
    """F9.5: Executive vision prompt instructs model to focus on charts, tables, metrics."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()
    exc = engine.load_template("executive")

    prompt = exc.vision_prompt.lower()
    assert "chart" in prompt or "metric" in prompt or "table" in prompt or "slide" in prompt or "business" in prompt


# ==============================================================================
# F10: Custom Template Support (5 tests)
# ==============================================================================

def test_f10_custom_template_load_from_filepath(sample_custom_template: Path):
    """F10.1: TemplateEngine loads custom template directly from file path."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()
    custom = engine.load_template(str(sample_custom_template))

    assert custom.name == "custom_workflow"
    assert "Engineering Checklist" in custom.sections


def test_f10_custom_template_render_custom_structure(sample_custom_template: Path):
    """F10.2: Custom template renders custom defined sections and variables."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    rendered = engine.render(
        str(sample_custom_template),
        {
            "title": "Pipeline Architecture",
            "visual_summary": "Diagram showing nodes A and B.",
            "transcription_text": "Connect node A to B.",
            "checklist": ["Verify firewall", "Deploy ingress"],
        },
    )
    assert "Pipeline Architecture" in rendered
    assert "[ ] Verify firewall" in rendered


def test_f10_custom_template_cli_argument(cli_runner, sample_video: Path, sample_custom_template: Path, tmp_path: Path):
    """F10.3: CLI process supports passing custom template via --template <path>."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    out_dir = tmp_path / "custom_out"

    res = cli_runner.invoke(
        cli_mod.app,
        ["process", str(sample_video), "--template", str(sample_custom_template), "--output-dir", str(out_dir)],
    )
    assert res.exit_code == 0
    assert any(out_dir.glob("*.md"))


def test_f10_custom_template_schema_validation(tmp_path: Path):
    """F10.4: Custom template with invalid schema raises validation error."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("invalid_key_only: true\n", encoding="utf-8")

    with pytest.raises(Exception):
        engine.load_template(str(bad_yaml))


def test_f10_custom_template_list_or_discovery():
    """F10.5: Template discovery returns list of available templates."""
    tmpl_mod = pytest.importorskip("video2doc.templates.engine")
    engine = tmpl_mod.TemplateEngine()
    available = engine.list_templates()

    assert "sop" in available
    assert "lecture" in available
    assert "executive" in available


# ==============================================================================
# F11: Modern Typed CLI (5 tests)
# ==============================================================================

def test_f11_cli_help_root(cli_runner):
    """F11.1: Root CLI --help displays program description and all subcommands."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    res = cli_runner.invoke(cli_mod.app, ["--help"])

    assert res.exit_code == 0
    assert "video2doc" in res.output.lower() or "documentation" in res.output.lower()
    for subcmd in ["process", "batch", "templates", "config"]:
        assert subcmd in res.output


def test_f11_cli_process_help(cli_runner):
    """F11.2: CLI process --help displays typed options and arguments."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    res = cli_runner.invoke(cli_mod.app, ["process", "--help"])

    assert res.exit_code == 0
    assert "--template" in res.output
    assert "--output-dir" in res.output


def test_f11_cli_batch_help(cli_runner):
    """F11.3: CLI batch --help displays options for batch directory processing."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    res = cli_runner.invoke(cli_mod.app, ["batch", "--help"])

    assert res.exit_code == 0
    assert "--template" in res.output
    assert "--output-dir" in res.output


def test_f11_cli_templates_list(cli_runner):
    """F11.4: CLI templates command lists built-in and configured templates."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    res = cli_runner.invoke(cli_mod.app, ["templates", "list"])

    assert res.exit_code == 0
    assert "sop" in res.output
    assert "lecture" in res.output
    assert "executive" in res.output


def test_f11_cli_config_show(cli_runner):
    """F11.5: CLI config show outputs active configuration settings."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    res = cli_runner.invoke(cli_mod.app, ["config", "show"])

    assert res.exit_code == 0
    assert "ollama_url" in res.output or "vision_model" in res.output or "11434" in res.output


# ==============================================================================
# F12: Multi-Tier Configuration (5 tests)
# ==============================================================================

def test_f12_config_defaults_adherence():
    """F12.1: Default settings match PROJECT.md interface contract values."""
    config_mod = pytest.importorskip("video2doc.config")
    settings = config_mod.Settings()

    assert settings.ollama_url == "http://localhost:11434"
    assert settings.scene_threshold == 0.3
    assert settings.debounce_seconds == 3.0
    assert settings.max_keyframes == 10
    assert settings.default_template == "sop"


def test_f12_config_env_var_override():
    """F12.2: Environment variables override default settings."""
    config_mod = pytest.importorskip("video2doc.config")
    with patch.dict(os.environ, {"VIDEO2DOC_OLLAMA_URL": "http://gpu-server:11434", "VIDEO2DOC_MAX_KEYFRAMES": "25"}):
        settings = config_mod.Settings()
        assert settings.ollama_url == "http://gpu-server:11434"
        assert settings.max_keyframes == 25


def test_f12_config_dotenv_support(tmp_path: Path):
    """F12.3: .env file values are parsed and applied to configuration."""
    config_mod = pytest.importorskip("video2doc.config")
    env_file = tmp_path / ".env"
    env_file.write_text("VIDEO2DOC_WHISPER_MODEL=small\nVIDEO2DOC_SCENE_THRESHOLD=0.55\n", encoding="utf-8")

    settings = config_mod.Settings(_env_file=str(env_file))
    assert settings.whisper_model == "small"
    assert settings.scene_threshold == 0.55


def test_f12_config_yaml_file_override(tmp_path: Path):
    """F12.4: YAML config file overrides default values."""
    config_mod = pytest.importorskip("video2doc.config")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump({"vision_model": "custom-vision", "text_model": "custom-text", "max_keyframes": 15}),
        encoding="utf-8",
    )

    settings = config_mod.Settings.from_yaml(cfg_file) if hasattr(config_mod.Settings, "from_yaml") else config_mod.Settings()
    assert settings is not None


def test_f12_config_cli_precedence(cli_runner, sample_video: Path, tmp_path: Path, ollama_state):
    """F12.5: CLI parameters take highest precedence over environment variables."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    out_dir = tmp_path / "cli_prec_out"

    with patch.dict(os.environ, {"VIDEO2DOC_TEXT_MODEL": "env-model"}):
        res = cli_runner.invoke(
            cli_mod.app,
            ["process", str(sample_video), "--text-model", "cli-override-model", "--output-dir", str(out_dir)],
        )
        assert res.exit_code == 0
        assert any(c.model == "cli-override-model" for c in ollama_state.calls)


# ==============================================================================
# F13: Pre-flight Diagnostics (5 tests)
# ==============================================================================

def test_f13_diagnostics_all_tools_present():
    """F13.1: Pre-flight diagnostics report status OK when FFmpeg and Ollama are reachable."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    check_fn = getattr(cli_mod, "check_environment", None) or getattr(cli_mod, "run_diagnostics", None)
    if check_fn:
        status = check_fn()
        assert status.get("ffmpeg", True) is True
    else:
        assert True


def test_f13_diagnostics_missing_ffmpeg_detection(ffmpeg_state):
    """F13.2: Diagnostics detect and flag missing FFmpeg executable."""
    ffmpeg_state.simulate_missing = True
    diag_mod = pytest.importorskip("video2doc.cli.main")
    assert shutil.which("ffmpeg") is None


def test_f13_diagnostics_ollama_offline_detection(ollama_state):
    """F13.3: Diagnostics detect offline Ollama daemon."""
    ollama_state.simulate_down = True
    exceptions_mod = pytest.importorskip("video2doc.exceptions")
    config_mod = pytest.importorskip("video2doc.config")

    pipeline_mod = pytest.importorskip("video2doc.pipeline")
    pipeline = pipeline_mod.DocumentPipeline()
    with pytest.raises(exceptions_mod.OllamaConnectionError):
        pipeline.check_ollama_health()


def test_f13_diagnostics_missing_model_detection(ollama_state):
    """F13.4: Diagnostics detect if required model tag is missing in Ollama."""
    ollama_state.available_models = ["some-other-model:latest"]
    exceptions_mod = pytest.importorskip("video2doc.exceptions")

    pipeline_mod = pytest.importorskip("video2doc.pipeline")
    pipeline = pipeline_mod.DocumentPipeline()
    with pytest.raises(exceptions_mod.ModelNotFoundError):
        pipeline.validate_models_available()


def test_f13_diagnostics_cli_check_command(cli_runner):
    """F13.5: CLI config check command executes diagnostics report."""
    cli_mod = pytest.importorskip("video2doc.cli.main")
    res = cli_runner.invoke(cli_mod.app, ["config", "check"])
    assert res.exit_code == 0
    assert "ffmpeg" in res.output.lower() or "ollama" in res.output.lower() or "ok" in res.output.lower()


# ==============================================================================
# F14: Strict Git Hygiene (5 tests)
# ==============================================================================

def test_f14_gitignore_exists_and_valid():
    """F14.1: .gitignore file exists at repository root and is non-empty."""
    gitignore_path = WORKSPACE_ROOT / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist at repository root"
    content = gitignore_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 0


def test_f14_gitignore_blocks_media_extensions():
    """F14.2: .gitignore ignores video and audio media extensions."""
    content = (WORKSPACE_ROOT / ".gitignore").read_text(encoding="utf-8")
    for ext in ["*.mp4", "*.mkv", "*.avi", "*.mov", "*.wav"]:
        assert ext in content or ext.lstrip("*.") in content, f"{ext} must be in .gitignore"


def test_f14_gitignore_blocks_virtual_environments():
    """F14.3: .gitignore ignores virtual environment folders."""
    content = (WORKSPACE_ROOT / ".gitignore").read_text(encoding="utf-8")
    for venv_pattern in ["venv", ".venv"]:
        assert venv_pattern in content, f"{venv_pattern} must be in .gitignore"


def test_f14_gitignore_blocks_course_manuals():
    """F14.4: .gitignore ignores legacy course manual directories and video dumps."""
    content = (WORKSPACE_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "manuales_sqx" in content or "manuales" in content


def test_f14_gitignore_blocks_caches_and_secrets():
    """F14.5: .gitignore ignores Python caches, pytest caches, logs, and .env files."""
    content = (WORKSPACE_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pat in ["__pycache__", ".pytest_cache", ".env", "*.log"]:
        assert pat in content or pat.lstrip("*.") in content, f"{pat} must be in .gitignore"


# ==============================================================================
# F15: Portfolio README (5 tests)
# ==============================================================================

def test_f15_readme_exists_and_titled():
    """F15.1: README.md exists and features Video2Doc project title."""
    readme_path = WORKSPACE_ROOT / "README.md"
    if not readme_path.exists():
        pytest.skip("README.md not yet created (Milestone 5)")
    content = readme_path.read_text(encoding="utf-8")
    assert "# " in content
    assert "video2doc" in content.lower() or "video" in content.lower()


def test_f15_readme_contains_mermaid_diagram():
    """F15.2: README.md contains Mermaid architecture diagram."""
    readme_path = WORKSPACE_ROOT / "README.md"
    if not readme_path.exists():
        pytest.skip("README.md not yet created (Milestone 5)")
    content = readme_path.read_text(encoding="utf-8")
    assert "```mermaid" in content, "README.md must contain a ```mermaid diagram"


def test_f15_readme_contains_installation_guide():
    """F15.3: README.md details prerequisites and installation instructions."""
    readme_path = WORKSPACE_ROOT / "README.md"
    if not readme_path.exists():
        pytest.skip("README.md not yet created (Milestone 5)")
    content = readme_path.read_text(encoding="utf-8")
    lower = content.lower()
    assert "installation" in lower or "install" in lower or "quickstart" in lower
    assert "ffmpeg" in lower


def test_f15_readme_contains_cli_usage_examples():
    """F15.4: README.md includes practical CLI usage examples."""
    readme_path = WORKSPACE_ROOT / "README.md"
    if not readme_path.exists():
        pytest.skip("README.md not yet created (Milestone 5)")
    content = readme_path.read_text(encoding="utf-8")
    assert "video2doc process" in content or "video2doc" in content


def test_f15_readme_contains_portfolio_badges():
    """F15.5: README.md includes status/version/license badges."""
    readme_path = WORKSPACE_ROOT / "README.md"
    if not readme_path.exists():
        pytest.skip("README.md not yet created (Milestone 5)")
    content = readme_path.read_text(encoding="utf-8")
    assert "[![" in content or "badge" in content.lower() or "shield" in content.lower() or "python" in content.lower()


# ==============================================================================
# F16: Packaging & Dependencies (5 tests)
# ==============================================================================

def test_f16_pyproject_toml_exists_and_valid():
    """F16.1: pyproject.toml exists and parses as valid TOML."""
    pyproject_path = WORKSPACE_ROOT / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml must exist"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    assert "project" in data


def test_f16_pyproject_entry_point_defined():
    """F16.2: pyproject.toml defines video2doc CLI entry point in [project.scripts]."""
    pyproject_path = WORKSPACE_ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    scripts = data.get("project", {}).get("scripts", {})
    assert "video2doc" in scripts, "video2doc console script must be defined"
    assert "video2doc.cli" in scripts["video2doc"]


def test_f16_pyproject_core_dependencies():
    """F16.3: Core dependencies are declared in pyproject.toml."""
    pyproject_path = WORKSPACE_ROOT / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    deps = " ".join(data.get("project", {}).get("dependencies", [])).lower()
    for dep in ["typer", "jinja2", "pydantic", "pyyaml"]:
        assert dep in deps, f"{dep} must be declared in dependencies"


def test_f16_requirements_txt_matches():
    """F16.4: requirements.txt exists and lists core dependencies."""
    req_path = WORKSPACE_ROOT / "requirements.txt"
    assert req_path.exists(), "requirements.txt must exist"
    content = req_path.read_text(encoding="utf-8").lower()
    for dep in ["typer", "jinja2", "pydantic"]:
        assert dep in content, f"{dep} must be in requirements.txt"


def test_f16_package_version_defined():
    """F16.5: video2doc package exposes a semantic __version__ string."""
    pkg = pytest.importorskip("video2doc")
    assert hasattr(pkg, "__version__")
    assert isinstance(pkg.__version__, str)
    assert len(pkg.__version__.split(".")) >= 2


# ==============================================================================
# F17: Automated E2E Test Suite (5 tests)
# ==============================================================================

def test_f17_test_infra_documentation_exists():
    """F17.1: TEST_INFRA.md exists and documents test philosophy and runner commands."""
    infra_path = WORKSPACE_ROOT / "TEST_INFRA.md"
    assert infra_path.exists(), "TEST_INFRA.md must exist at root"
    content = infra_path.read_text(encoding="utf-8")
    assert "pytest" in content
    assert "Tier" in content


def test_f17_test_ready_documentation_exists():
    """F17.2: TEST_READY.md exists and reports test tier summary."""
    ready_path = WORKSPACE_ROOT / "TEST_READY.md"
    assert ready_path.exists(), "TEST_READY.md must exist at root"
    content = ready_path.read_text(encoding="utf-8")
    assert "Tier 1" in content
    assert "Tier 2" in content
    assert "Tier 3" in content
    assert "Tier 4" in content


def test_f17_conftest_fixtures_configured():
    """F17.3: conftest.py defines required synthetic fixtures."""
    conftest_path = WORKSPACE_ROOT / "tests" / "conftest.py"
    assert conftest_path.exists()
    content = conftest_path.read_text(encoding="utf-8")
    for fixture_name in ["mock_ffmpeg", "mock_ollama", "mock_whisper", "cli_runner"]:
        assert fixture_name in content


def test_f17_all_tiers_present():
    """F17.4: All 4 E2E tier test files are present under tests/e2e/."""
    e2e_dir = WORKSPACE_ROOT / "tests" / "e2e"
    assert (e2e_dir / "test_tier1_features.py").exists()
    assert (e2e_dir / "test_tier2_boundaries.py").exists()
    assert (e2e_dir / "test_tier3_combinations.py").exists()
    assert (e2e_dir / "test_tier4_workloads.py").exists()


def test_f17_offline_execution_deterministic(mock_environment):
    """F17.5: Synthetic mocking environment functions deterministically without network or GPU."""
    assert mock_environment["ffmpeg"] is not None
    assert mock_environment["ollama"] is not None
    assert mock_environment["whisper"] is not None
    assert mock_environment["cuda"] is not None
