"""Comprehensive unit tests for Video2Doc Domain-Agnostic Template Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import pytest
from pydantic import ValidationError

from video2doc.exceptions import TemplateNotFoundError
from video2doc.templates import (
    TemplateDefinition,
    TemplateEngine,
    TemplateLoader,
)
from video2doc.templates.engine import (
    code_block,
    format_duration,
    format_timestamp,
    markdown_clean,
    relative_img,
    truncate_chars,
)


# ==============================================================================
# 1. TemplateDefinition Schema Tests
# ==============================================================================

class TestTemplateDefinitionSchema:
    def test_valid_template_definition(self):
        tmpl = TemplateDefinition(
            name="test_template",
            description="A test template",
            version="1.0",
            vision_prompt="Examine visual features",
            synthesis_prompt="Synthesize document",
            sections=["Section 1", "Section 2"],
            layout_template="# {{ title }}",
        )
        assert tmpl.name == "test_template"
        assert tmpl.description == "A test template"
        assert tmpl.version == "1.0"
        assert tmpl.vision_prompt == "Examine visual features"
        assert tmpl.synthesis_prompt == "Synthesize document"
        assert tmpl.sections == ["Section 1", "Section 2"]
        assert tmpl.layout_template == "# {{ title }}"

    def test_default_values(self):
        tmpl = TemplateDefinition(
            name="minimal",
            description="Minimal template",
            vision_prompt="Look at images",
            synthesis_prompt="Write document",
        )
        assert tmpl.version == "1.0"
        assert tmpl.sections == []
        assert tmpl.layout_template == ""

    def test_missing_required_fields_raises_validation_error(self):
        with pytest.raises(ValidationError):
            TemplateDefinition.model_validate({"name": "incomplete"})

        with pytest.raises(ValidationError):
            TemplateDefinition.model_validate({
                "name": "no_prompts",
                "description": "missing vision and synthesis",
            })

    def test_nested_format_normalization(self):
        nested_data = {
            "metadata": {
                "name": "nested_tmpl",
                "description": "Nested format template",
                "version": "2.0.0",
            },
            "vision": {
                "user_prompt": "Inspect slides and formulas",
                "system_instruction": "You are a vision assistant",
            },
            "synthesis": {
                "synthesis_instructions": "Synthesize notes",
                "output_schema": "# Document Schema",
            },
            "sections": ["Overview", "Derivations"],
        }
        tmpl = TemplateDefinition.model_validate(nested_data)
        assert tmpl.name == "nested_tmpl"
        assert tmpl.description == "Nested format template"
        assert tmpl.version == "2.0.0"
        assert tmpl.vision_prompt == "Inspect slides and formulas"
        assert tmpl.synthesis_prompt == "Synthesize notes"
        assert tmpl.sections == ["Overview", "Derivations"]
        assert tmpl.layout_template == "# Document Schema"


# ==============================================================================
# 2. TemplateLoader Unit Tests
# ==============================================================================

class TestTemplateLoader:
    def test_load_builtin_sop(self):
        loader = TemplateLoader()
        sop = loader.load("sop")
        assert sop.name == "sop"
        assert bool(sop.vision_prompt)
        assert bool(sop.synthesis_prompt)
        assert len(sop.sections) > 0

    def test_load_builtin_lecture(self):
        loader = TemplateLoader()
        lecture = loader.load("lecture")
        assert lecture.name == "lecture"
        assert bool(lecture.vision_prompt)
        assert bool(lecture.synthesis_prompt)
        assert len(lecture.sections) > 0

    def test_load_builtin_executive(self):
        loader = TemplateLoader()
        executive = loader.load("executive")
        assert executive.name == "executive"
        assert bool(executive.vision_prompt)
        assert bool(executive.synthesis_prompt)
        assert len(executive.sections) > 0

    def test_load_builtin_with_yaml_extension(self):
        loader = TemplateLoader()
        sop = loader.load("sop.yaml")
        assert sop.name == "sop"

    def test_load_custom_template_from_file_path(self, tmp_path: Path):
        custom_file = tmp_path / "custom.yaml"
        custom_file.write_text(
            """name: custom_guide
description: Custom user guide
vision_prompt: Analyze UI steps
synthesis_prompt: Write custom SOP
sections:
  - Intro
  - Procedure
""",
            encoding="utf-8",
        )
        loader = TemplateLoader()
        tmpl = loader.load(str(custom_file))
        assert tmpl.name == "custom_guide"
        assert tmpl.sections == ["Intro", "Procedure"]

    def test_load_nonexistent_template_raises_template_not_found(self):
        loader = TemplateLoader()
        with pytest.raises(TemplateNotFoundError) as exc_info:
            loader.load("nonexistent_unknown_template")
        assert "nonexistent_unknown_template" in str(exc_info.value)
        assert len(exc_info.value.searched_paths) > 0

    def test_load_invalid_yaml_syntax_raises_template_not_found(self, tmp_path: Path):
        broken = tmp_path / "broken.yaml"
        broken.write_text("name: broken\n  bad_indent: [unclosed\n", encoding="utf-8")
        loader = TemplateLoader()
        with pytest.raises(TemplateNotFoundError) as exc_info:
            loader.load(str(broken))
        err_msg = str(exc_info.value).lower()
        assert "yaml" in err_msg or "syntax" in err_msg

    def test_load_non_dict_yaml_raises_template_not_found(self, tmp_path: Path):
        plain_string_yaml = tmp_path / "scalar.yaml"
        plain_string_yaml.write_text("just a plain string", encoding="utf-8")
        loader = TemplateLoader()
        with pytest.raises(TemplateNotFoundError) as exc_info:
            loader.load(str(plain_string_yaml))
        assert "yaml" in str(exc_info.value).lower() or "syntax" in str(exc_info.value).lower()

    def test_load_missing_fields_yaml_raises_validation_error(self, tmp_path: Path):
        incomplete = tmp_path / "incomplete.yaml"
        incomplete.write_text("name: incomplete\ndescription: missing prompts\n", encoding="utf-8")
        loader = TemplateLoader()
        with pytest.raises(ValidationError):
            loader.load(str(incomplete))

    def test_list_builtin_names(self):
        loader = TemplateLoader()
        names = loader.list_builtin_names()
        assert "sop" in names
        assert "lecture" in names
        assert "executive" in names

    def test_list_available_templates(self, tmp_path: Path):
        custom_dir = tmp_path / "user_templates"
        custom_dir.mkdir()
        (custom_dir / "my_custom.yaml").write_text(
            """name: my_custom
description: My custom template
vision_prompt: Check visuals
synthesis_prompt: Write doc
""",
            encoding="utf-8",
        )
        loader = TemplateLoader(custom_dirs=[custom_dir])
        available = loader.list_available_templates()
        names = [t["name"] for t in available]
        assert "sop" in names
        assert "lecture" in names
        assert "executive" in names
        assert "my_custom" in names


# ==============================================================================
# 3. Built-in Templates Content Verification Tests
# ==============================================================================

class TestBuiltinTemplatesContent:
    def test_sop_template_structure_and_prompts(self):
        engine = TemplateEngine()
        sop = engine.load_template("sop")

        assert sop.name == "sop"
        assert "1.0" in sop.version

        # Vision prompt checks: UI, clicks, parameters, screenshot
        v_prompt = sop.vision_prompt.lower()
        assert any(k in v_prompt for k in ["ui", "interface", "click", "screenshot", "parameter"])

        # Sections check: procedure, prerequisites, steps
        sections_str = " ".join(sop.sections).lower()
        assert any(k in sections_str for k in ["prerequisite", "step", "procedure", "settings"])

        # Layout template check
        assert bool(sop.layout_template)
        assert "title" in sop.layout_template

    def test_lecture_template_structure_and_prompts(self):
        engine = TemplateEngine()
        lecture = engine.load_template("lecture")

        assert lecture.name == "lecture"

        # Vision prompt checks: slides, formulas, diagrams, LaTeX
        v_prompt = lecture.vision_prompt.lower()
        assert any(k in v_prompt for k in ["slide", "diagram", "formula", "latex", "visual"])

        # Sections check: objective, concept, summary, derivation
        sections_str = " ".join(lecture.sections).lower()
        assert any(k in sections_str for k in ["objective", "concept", "derivation", "summary", "citation"])

        # Layout template check
        assert bool(lecture.layout_template)

    def test_executive_template_structure_and_prompts(self):
        engine = TemplateEngine()
        executive = engine.load_template("executive")

        assert executive.name == "executive"

        # Vision prompt checks: metrics, KPIs, charts, tables
        v_prompt = executive.vision_prompt.lower()
        assert any(k in v_prompt for k in ["chart", "metric", "kpi", "table", "slide", "business"])

        # Sections check: summary, decision, action items, KPIs
        sections_str = " ".join(executive.sections).lower()
        assert any(k in sections_str for k in ["summary", "decision", "action", "kpi", "metric"])

        # Layout template check
        assert bool(executive.layout_template)


# ==============================================================================
# 4. Custom Jinja2 Filters Unit Tests
# ==============================================================================

class TestCustomFilters:
    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (0, "00:00"),
            (5, "00:05"),
            (45, "00:45"),
            (60, "01:00"),
            (125, "02:05"),
            (125.7, "02:05"),
            (3600, "01:00:00"),
            (3661, "01:01:01"),
            (7325, "02:02:05"),
            (None, "00:00"),
            ("", "00:00"),
            (-15, "00:00"),
            ("invalid", "00:00"),
        ],
    )
    def test_format_timestamp(self, seconds: Any, expected: str):
        assert format_timestamp(seconds) == expected

    @pytest.mark.parametrize(
        "seconds, expected",
        [
            (0, "0s"),
            (45, "45s"),
            (60, "1m 0s"),
            (125, "2m 5s"),
            (3665, "1h 1m 5s"),
            (None, "0s"),
            ("", "0s"),
            (-5, "0s"),
            ("invalid", "0s"),
        ],
    )
    def test_format_duration(self, seconds: Any, expected: str):
        assert format_duration(seconds) == expected

    def test_markdown_clean(self):
        raw = "  \n\n# Header\n\n\n\n\nParagraph text   \n\n"
        cleaned = markdown_clean(raw)
        assert cleaned.startswith("# Header")
        assert cleaned.endswith("Paragraph text")
        assert "\n\n\n" not in cleaned
        assert markdown_clean(None) == ""

    def test_code_block(self):
        res = code_block("x = 10\nprint(x)", language="python")
        assert res == "```python\nx = 10\nprint(x)\n```"

        res_no_lang = code_block("echo hello")
        assert res_no_lang == "```\necho hello\n```"

        assert code_block(None) == "```\n\n```"

    def test_relative_img(self):
        assert relative_img("frames/001.jpg") == "frames/001.jpg"
        assert relative_img(Path("frames") / "001.jpg") == "frames/001.jpg"
        assert relative_img(None) == ""

    def test_truncate_chars(self):
        text = "Hello world this is a test string"
        assert truncate_chars(text, max_len=15) == "Hello world ..."
        assert truncate_chars(text, max_len=100) == text
        assert truncate_chars(None, max_len=10) == ""


# ==============================================================================
# 5. TemplateEngine Rendering Unit Tests
# ==============================================================================

class TestTemplateEngineRendering:
    def test_render_string_expressions_and_loops(self):
        engine = TemplateEngine()
        template = "Title: {{ title }} | Items: {% for i in items %}{{ i }}{% if not loop.last %},{% endif %}{% endfor %}"
        context = {"title": "Test Run", "items": ["alpha", "beta", "gamma"]}
        out = engine.render_string(template, context)
        assert out == "Title: Test Run | Items: alpha,beta,gamma"

    def test_render_string_custom_filters(self):
        engine = TemplateEngine()
        template = "Timestamp: {{ 125 | format_timestamp }} | Duration: {{ 125 | format_duration }}"
        out = engine.render_string(template, {})
        assert "Timestamp: 02:05" in out
        assert "Duration: 2m 5s" in out

    def test_render_string_code_block_filter(self):
        engine = TemplateEngine()
        template = "{{ 'import sys' | code_block('python') }}"
        out = engine.render_string(template, {})
        assert "```python" in out
        assert "import sys" in out

    def test_render_missing_variables_graceful(self):
        engine = TemplateEngine()
        template = "Name: {{ name | default('Anonymous') }} | Stem: {{ video.stem | default('no_video') }}"
        out = engine.render_string(template, {})
        assert "Name: Anonymous" in out
        assert "Stem: no_video" in out

    def test_render_builtin_sop_with_context(self):
        engine = TemplateEngine()
        context = {
            "title": "Software Setup Guide",
            "transcription_text": "First install dependencies then run script.",
            "visual_summary": "Dashboard interface screenshot.",
            "keyframes": [
                {"timestamp": 5.0, "path": "frame_1.jpg", "description": "Settings UI"},
                {"timestamp": 25.0, "path": "frame_2.jpg", "description": "Success Dialog"},
            ],
        }
        rendered = engine.render("sop", context)
        assert "Software Setup Guide" in rendered
        assert "Step-by-Step Procedure" in rendered
        assert "00:05" in rendered
        assert "00:25" in rendered
        assert "Settings UI" in rendered

    def test_render_builtin_lecture_with_context(self):
        engine = TemplateEngine()
        context = {
            "title": "Machine Learning Fundamentals",
            "transcription_text": "Gradient descent minimizes the convex loss function.",
            "visual_summary": "Slide showing cost curve and equations.",
            "formulas": ["\\nabla f(x) = 0", "w_{t+1} = w_t - \\eta \\nabla L"],
        }
        rendered = engine.render("lecture", context)
        assert "Machine Learning Fundamentals" in rendered
        assert "Gradient descent minimizes" in rendered
        assert "\\nabla f(x) = 0" in rendered

    def test_render_builtin_executive_with_context(self):
        engine = TemplateEngine()
        context = {
            "title": "Q3 Strategic Review",
            "transcription_text": "Revenue grew by 24% and expansion into EMEA is approved.",
            "visual_summary": "Revenue chart slide.",
            "action_items": [
                {"task": "Deploy cloud infrastructure", "owner": "DevOps Lead", "due": "Q4", "priority": "High"},
            ],
            "kpis": [
                {"name": "ARR Growth", "target": "25%", "actual": "24%", "variance": "-1%"},
            ],
        }
        rendered = engine.render("executive", context)
        assert "Q3 Strategic Review" in rendered
        assert "Deploy cloud infrastructure" in rendered
        assert "DevOps Lead" in rendered
        assert "ARR Growth" in rendered

    def test_render_custom_template_path(self, tmp_path: Path):
        tmpl_file = tmp_path / "custom_workflow.yaml"
        tmpl_file.write_text(
            """name: custom_workflow
description: Custom engineering workflow template
vision_prompt: Analyze diagram
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
""",
            encoding="utf-8",
        )
        engine = TemplateEngine()
        rendered = engine.render(
            str(tmpl_file),
            {
                "title": "Pipeline Architecture",
                "visual_summary": "Diagram showing nodes A and B.",
                "transcription_text": "Connect node A to B.",
                "checklist": ["Verify firewall", "Deploy ingress"],
            },
        )
        assert "Pipeline Architecture" in rendered
        assert "Diagram showing nodes A and B." in rendered
        assert "[ ] Verify firewall" in rendered
        assert "[ ] Deploy ingress" in rendered

    def test_render_pydantic_context_model(self):
        class DummyContext(TemplateDefinition):
            pass

        ctx_model = DummyContext(
            name="pydantic_ctx",
            description="Testing pydantic model context dump",
            vision_prompt="V",
            synthesis_prompt="S",
        )
        engine = TemplateEngine()
        rendered = engine.render_string("Name: {{ name }} | Desc: {{ description }}", ctx_model)  # type: ignore[arg-type]
        assert "Name: pydantic_ctx" in rendered
        assert "Desc: Testing pydantic model context dump" in rendered

    def test_list_templates(self):
        engine = TemplateEngine()
        available = engine.list_templates()
        assert "sop" in available
        assert "lecture" in available
        assert "executive" in available

    def test_list_available_templates(self):
        engine = TemplateEngine()
        available_meta = engine.list_available_templates()
        assert isinstance(available_meta, list)
        names = [item["name"] for item in available_meta]
        assert "sop" in names
        assert "lecture" in names
        assert "executive" in names


# ==============================================================================
# 6. Package Module Exports Tests
# ==============================================================================

def test_package_exports():
    import video2doc.templates as tmpl_pkg

    assert hasattr(tmpl_pkg, "TemplateDefinition")
    assert hasattr(tmpl_pkg, "TemplateLoader")
    assert hasattr(tmpl_pkg, "TemplateEngine")
    assert hasattr(tmpl_pkg, "TemplateNotFoundError")
    assert issubclass(tmpl_pkg.TemplateNotFoundError, Exception)
