"""Domain-agnostic template rendering engine using Jinja2."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import jinja2

from video2doc.templates.loader import TemplateDefinition, TemplateLoader


def format_timestamp(seconds: float | int | str | None) -> str:
    """Format numeric seconds into MM:SS or HH:MM:SS format."""
    if seconds is None or seconds == "":
        return "00:00"
    try:
        total = int(float(seconds))
    except (ValueError, TypeError):
        return "00:00"

    if total < 0:
        total = 0

    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_duration(seconds: float | int | str | None) -> str:
    """Format numeric seconds into human-readable duration string (e.g. '2m 5s' or '1h 2m 5s')."""
    if seconds is None or seconds == "":
        return "0s"
    try:
        total = int(float(seconds))
    except (ValueError, TypeError):
        return "0s"

    if total < 0:
        total = 0

    hours = total // 3600
    mins = (total % 3600) // 60
    secs = total % 60

    parts: list[str] = []
    if hours > 0:
        parts.append(f"{hours}h")
    if mins > 0 or hours > 0:
        parts.append(f"{mins}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def markdown_clean(text: Any) -> str:
    """Clean, strip trailing whitespace, and normalize excessive blank lines."""
    if text is None:
        return ""
    cleaned = str(text).strip()
    return re.sub(r"\n{3,}", "\n\n", cleaned)


def code_block(code: Any, language: str = "") -> str:
    """Wrap code string in markdown code fences."""
    code_str = str(code) if code is not None else ""
    lang = str(language).strip()
    return f"```{lang}\n{code_str}\n```"


def relative_img(path: Any, doc_dir: Any = "") -> str:
    """Normalize relative image path for markdown document embedding."""
    if not path:
        return ""
    return Path(str(path)).as_posix()


def truncate_chars(text: Any, max_len: int = 100) -> str:
    """Truncate text safely with ellipsis if exceeding max length."""
    s = str(text or "")
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 3)] + "..."


class TemplateEngine:
    """Domain-agnostic template engine providing Jinja2 rendering and template management."""

    def __init__(self, loader: TemplateLoader | None = None) -> None:
        self.loader = loader or TemplateLoader()
        self.env = jinja2.Environment(
            autoescape=False,
            undefined=jinja2.ChainableUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._register_filters()

    def _register_filters(self) -> None:
        """Register custom filters for timestamps, durations, code blocks, and markdown cleanup."""
        self.env.filters["format_timestamp"] = format_timestamp
        self.env.filters["format_duration"] = format_duration
        self.env.filters["markdown_clean"] = markdown_clean
        self.env.filters["code_block"] = code_block
        self.env.filters["relative_img"] = relative_img
        self.env.filters["truncate_chars"] = truncate_chars

    def load_template(self, name_or_path: str | Path) -> TemplateDefinition:
        """Load and validate a template definition by name or file path."""
        return self.loader.load(name_or_path)

    def render_string(self, template_str: str, context: dict[str, Any] | None = None) -> str:
        """Render an arbitrary Jinja2 template string directly with given context."""
        if hasattr(context, "model_dump"):
            ctx = context.model_dump()  # type: ignore[union-attr]
        elif hasattr(context, "dict"):
            ctx = context.dict()  # type: ignore[union-attr]
        else:
            ctx = dict(context or {})
        template = self.env.from_string(template_str)
        return template.render(**ctx)

    def render(self, template_name_or_path: str | Path | TemplateDefinition, context: dict[str, Any] | None = None) -> str:
        """Load template and render its layout (or synthesis prompt) with context data."""
        if isinstance(template_name_or_path, TemplateDefinition):
            template_def = template_name_or_path
        else:
            template_def = self.load_template(template_name_or_path)

        # Convert context if a Pydantic model is supplied
        if hasattr(context, "model_dump"):
            ctx = context.model_dump()  # type: ignore[union-attr]
        elif hasattr(context, "dict"):
            ctx = context.dict()  # type: ignore[union-attr]
        else:
            ctx = dict(context or {})

        ctx.setdefault("template", template_def)

        # Choose template body: layout_template if non-empty, else fallback to synthesis_prompt
        template_body = (
            template_def.layout_template.strip()
            if template_def.layout_template and template_def.layout_template.strip()
            else template_def.synthesis_prompt
        )

        template = self.env.from_string(template_body)
        return template.render(**ctx)

    def list_templates(self) -> list[str]:
        """Return names of available built-in templates."""
        return self.loader.list_builtin_names()

    def list_available_templates(self) -> list[dict[str, str]]:
        """Return rich metadata list for all discovered built-in and user templates."""
        return self.loader.list_available_templates()
