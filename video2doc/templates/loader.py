"""Template loader and schema validation for Video2Doc."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from video2doc.exceptions import TemplateNotFoundError


class TemplateDefinition(BaseModel):
    """Domain-agnostic template definition matching PROJECT.md interface contracts."""

    name: str = Field(..., description="Unique template identifier (e.g. 'sop', 'lecture', 'executive')")
    description: str = Field(..., description="Human-readable description of domain and usage")
    version: str = Field(default="1.0", description="Semantic version of template")
    vision_prompt: str = Field(..., description="Domain-specific instructions for keyframe visual analysis")
    synthesis_prompt: str = Field(..., description="Instructions and schema for final document text synthesis")
    sections: list[str] = Field(default_factory=list, description="List of standard section headings")
    layout_template: str = Field(default="", description="Optional Jinja2 Markdown layout template")

    @model_validator(mode="before")
    @classmethod
    def _normalize_nested_or_flat(cls, data: Any) -> Any:
        """Support both flat schemas and structured schemas (metadata, vision, synthesis)."""
        if not isinstance(data, dict):
            return data

        # Check if structured under metadata/vision/synthesis keys
        if "metadata" in data and isinstance(data["metadata"], dict) and "name" not in data:
            meta = data["metadata"]
            vis = data.get("vision", {})
            syn = data.get("synthesis", {})

            v_prompt = ""
            if isinstance(vis, dict):
                v_prompt = vis.get("user_prompt") or vis.get("system_instruction") or ""
            elif isinstance(vis, str):
                v_prompt = vis

            s_prompt = ""
            if isinstance(syn, dict):
                s_prompt = syn.get("synthesis_instructions") or syn.get("output_schema") or ""
            elif isinstance(syn, str):
                s_prompt = syn

            return {
                "name": meta.get("name", ""),
                "description": meta.get("description", ""),
                "version": meta.get("version", "1.0"),
                "vision_prompt": v_prompt,
                "synthesis_prompt": s_prompt,
                "sections": data.get("sections", []),
                "layout_template": data.get("markdown_layout") or (syn.get("output_schema") if isinstance(syn, dict) else ""),
            }

        return data


class TemplateLoader:
    """Discovers, loads, and validates built-in and user-defined YAML templates."""

    def __init__(self, custom_dirs: list[Path | str] | None = None) -> None:
        self.builtins_dir = Path(__file__).resolve().parent / "builtins"
        self.custom_dirs: list[Path] = [Path(d).resolve() for d in (custom_dirs or [])]
        user_dir = Path.home() / ".video2doc" / "templates"
        if user_dir not in self.custom_dirs:
            self.custom_dirs.append(user_dir)

    def load(self, name_or_path: str | Path) -> TemplateDefinition:
        """Resolve and load a template by built-in name or local file path."""
        path = self.resolve_template_path(name_or_path)
        return self.load_from_path(path)

    def resolve_template_path(self, name_or_path: str | Path) -> Path:
        """Locate template file across built-ins, exact file paths, and configured search directories."""
        raw_str = str(name_or_path).strip()
        candidate = Path(raw_str).expanduser()

        # 1. Direct path check (absolute or relative to current working directory)
        if candidate.is_file():
            return candidate.resolve()

        if candidate.with_suffix(".yaml").is_file():
            return candidate.with_suffix(".yaml").resolve()
        if candidate.with_suffix(".yml").is_file():
            return candidate.with_suffix(".yml").resolve()

        searched: list[str] = [str(candidate)]

        # 2. Built-in templates directory
        stem = Path(raw_str).stem
        for ext in (".yaml", ".yml"):
            builtin_path = self.builtins_dir / f"{stem}{ext}"
            searched.append(str(builtin_path))
            if builtin_path.is_file():
                return builtin_path.resolve()

        # 3. Custom search directories
        for sdir in self.custom_dirs:
            for ext in (".yaml", ".yml", ""):
                custom_candidate = sdir / f"{stem}{ext}" if ext else sdir / raw_str
                searched.append(str(custom_candidate))
                if custom_candidate.is_file():
                    return custom_candidate.resolve()

        raise TemplateNotFoundError(
            raw_str,
            searched_paths=searched,
            message=f"Template '{raw_str}' not found in built-ins or searched paths.",
        )

    def load_from_path(self, path: Path | str) -> TemplateDefinition:
        """Load, parse, and validate a TemplateDefinition from a YAML file."""
        resolved_path = Path(path).resolve()
        if not resolved_path.is_file():
            raise TemplateNotFoundError(
                str(resolved_path),
                searched_paths=[str(resolved_path)],
                message=f"Template file '{resolved_path}' does not exist.",
            )

        try:
            content = resolved_path.read_text(encoding="utf-8")
        except Exception as e:
            raise TemplateNotFoundError(
                str(resolved_path),
                searched_paths=[str(resolved_path)],
                message=f"Failed to read template file '{resolved_path}': {e}",
            ) from e

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as e:
            raise TemplateNotFoundError(
                str(resolved_path),
                searched_paths=[str(resolved_path)],
                message=f"Invalid YAML syntax in template '{resolved_path}': {e}",
            ) from e

        if not isinstance(data, dict):
            raise TemplateNotFoundError(
                str(resolved_path),
                searched_paths=[str(resolved_path)],
                message=f"Invalid YAML syntax in template '{resolved_path}': expected mapping dictionary, got {type(data).__name__}",
            )

        # Validate against TemplateDefinition Pydantic model
        return TemplateDefinition.model_validate(data)

    def list_builtin_names(self) -> list[str]:
        """Return sorted list of names for all built-in templates."""
        if not self.builtins_dir.is_dir():
            return []
        names: set[str] = set()
        for p in self.builtins_dir.glob("*.yaml"):
            names.add(p.stem)
        for p in self.builtins_dir.glob("*.yml"):
            names.add(p.stem)
        return sorted(names)

    def list_available_templates(self) -> list[dict[str, str]]:
        """Return list of metadata dictionaries for all discovered templates."""
        results: list[dict[str, str]] = []

        # Built-in templates
        for name in self.list_builtin_names():
            try:
                tmpl = self.load(name)
                results.append({
                    "name": tmpl.name,
                    "description": tmpl.description,
                    "version": tmpl.version,
                    "type": "builtin",
                })
            except Exception:
                results.append({
                    "name": name,
                    "description": "Built-in documentation template",
                    "version": "1.0",
                    "type": "builtin",
                })

        # Custom directory templates
        for sdir in self.custom_dirs:
            if sdir.is_dir():
                for p in sorted(sdir.glob("*.yaml")):
                    try:
                        tmpl = self.load_from_path(p)
                        results.append({
                            "name": tmpl.name,
                            "description": tmpl.description,
                            "version": tmpl.version,
                            "type": "custom",
                            "path": str(p),
                        })
                    except Exception:
                        continue

        return results
