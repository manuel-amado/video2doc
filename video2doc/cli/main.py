"""Command-line interface for Video2Doc using Typer and Rich."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import requests
import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from video2doc.config import Settings, get_settings
from video2doc.exceptions import (
    MediaCorruptError,
    MissingDependencyError,
    ModelNotFoundError,
    OllamaConnectionError,
    TemplateNotFoundError,
    Video2DocError,
)
from video2doc.pipeline import DocumentPipeline
from video2doc.templates.engine import TemplateEngine

logger = logging.getLogger(__name__)
console = Console(highlight=False)

app = typer.Typer(
    name="video2doc",
    help="Video2Doc: Production-Ready Multimodal Video Documentation Framework.",
    no_args_is_help=True,
    add_completion=False,
)

templates_app = typer.Typer(
    name="templates",
    help="Discover, inspect, and manage documentation templates.",
    no_args_is_help=True,
)

config_app = typer.Typer(
    name="config",
    help="Inspect active configuration and run pre-flight system diagnostics.",
    no_args_is_help=True,
)

app.add_typer(templates_app, name="templates")
app.add_typer(config_app, name="config")


def check_environment(settings: Settings | None = None) -> dict[str, Any]:
    """Run diagnostic checks on FFmpeg, ffprobe, Ollama server, and configured models."""
    cfg = settings or get_settings()
    ffmpeg_ok = bool(shutil.which("ffmpeg"))
    ffprobe_ok = bool(shutil.which("ffprobe"))
    ollama_ok = False
    vision_model_ok = False
    text_model_ok = False
    available_models: list[str] = []
    ollama_error: str | None = None

    try:
        url = f"{cfg.ollama_url.rstrip('/')}/api/tags"
        resp = requests.get(url, timeout=5.0)
        if resp.ok:
            ollama_ok = True
            data = resp.json()
            models_list = data.get("models", [])
            available_models = [
                m.get("name", "")
                for m in models_list
                if isinstance(m, dict) and "name" in m
            ]

            def _is_present(required: str) -> bool:
                req_clean = required.lower().strip()
                for av in available_models:
                    av_clean = av.lower().strip()
                    if (
                        av_clean == req_clean
                        or av_clean.split(":")[0] == req_clean.split(":")[0]
                        or av_clean.startswith(req_clean)
                        or req_clean.startswith(av_clean)
                    ):
                        return True
                return False

            vision_model_ok = _is_present(cfg.vision_model)
            text_model_ok = _is_present(cfg.text_model)
        else:
            ollama_error = f"HTTP {resp.status_code}"
    except Exception as err:
        ollama_error = str(err)

    all_ok = bool(
        ffmpeg_ok and ffprobe_ok and ollama_ok and vision_model_ok and text_model_ok
    )

    return {
        "ffmpeg": ffmpeg_ok,
        "ffprobe": ffprobe_ok,
        "ollama": ollama_ok,
        "vision_model": vision_model_ok,
        "text_model": text_model_ok,
        "available_models": available_models,
        "ollama_error": ollama_error,
        "all_ok": all_ok,
    }


# Alias for compatibility with external tests
run_diagnostics = check_environment


def _build_settings_overrides(
    output_path: Path | None = None,
    template: str | None = None,
    model_vision: str | None = None,
    model_text: str | None = None,
    whisper_model: str | None = None,
    whisper_device: str | None = None,
    language: str | None = None,
    scene_threshold: float | None = None,
    max_frames: int | None = None,
    debounce_seconds: float | None = None,
    keep_frames: bool | None = None,
    embed_images: bool | None = None,
    ollama_url: str | None = None,
) -> dict[str, Any]:
    """Build a dictionary of non-None configuration overrides."""
    overrides: dict[str, Any] = {}
    if output_path is not None:
        overrides["output_dir"] = output_path
    if template is not None:
        overrides["default_template"] = template
    if model_vision is not None:
        overrides["vision_model"] = model_vision
    if model_text is not None:
        overrides["text_model"] = model_text
    if whisper_model is not None:
        overrides["whisper_model"] = whisper_model
    if whisper_device is not None:
        overrides["whisper_device"] = whisper_device
    if language is not None:
        overrides["whisper_language"] = language
    if scene_threshold is not None:
        overrides["scene_threshold"] = scene_threshold
    if max_frames is not None:
        overrides["max_keyframes"] = max_frames
    if debounce_seconds is not None:
        overrides["debounce_seconds"] = debounce_seconds
    if keep_frames is not None:
        overrides["keep_frames"] = keep_frames
    if embed_images is not None:
        overrides["embed_images"] = embed_images
    if ollama_url is not None:
        overrides["ollama_url"] = ollama_url
    return overrides


def _handle_cli_error(exc: Exception, debug: bool = False) -> None:
    """Format and display domain or system errors in a Rich panel and exit."""
    if isinstance(exc, Video2DocError):
        details_str = ""
        if getattr(exc, "details", None):
            details_items = [f"• {k}: {escape(str(v))}" for k, v in exc.details.items() if v]
            if details_items:
                details_str = "\n\n[bold]Details:[/bold]\n" + "\n".join(details_items)

        console.print(
            Panel(
                f"[bold red]{exc.__class__.__name__}:[/bold red] {escape(exc.message)}{details_str}",
                title="[bold red]Video2Doc Error[/bold red]",
                border_style="red",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold red]Execution Error:[/bold red] {escape(str(exc))}",
                title="[bold red]Video2Doc Error[/bold red]",
                border_style="red",
            )
        )

    if debug:
        console.print_exception()
    raise typer.Exit(code=1)


@app.command("process")
def process_command(
    video: Optional[Path] = typer.Argument(
        None,
        help="Path to the input video file (.mp4, .mkv, .mov, .avi, .webm)",
        show_default=False,
    ),
    input: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to the input video file (alternative to positional argument)",
        show_default=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Target output directory or Markdown file destination",
        show_default=False,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Target output directory (alias for --output)",
        show_default=False,
    ),
    template: str = typer.Option(
        "sop",
        "--template",
        "-t",
        help="Template name ('sop', 'lecture', 'executive') or path to custom YAML template",
    ),
    model_vision: Optional[str] = typer.Option(
        None,
        "--model-vision",
        "--vision-model",
        help="Ollama vision model name (e.g. 'llama3.2-vision')",
        show_default=False,
    ),
    model_text: Optional[str] = typer.Option(
        None,
        "--model-text",
        "--text-model",
        help="Ollama text model name (e.g. 'qwen2.5-coder:14b')",
        show_default=False,
    ),
    whisper_model: Optional[str] = typer.Option(
        None,
        "--whisper-model",
        help="OpenAI Whisper model variant (tiny, base, small, medium, turbo)",
        show_default=False,
    ),
    whisper_device: Optional[str] = typer.Option(
        None,
        "--whisper-device",
        help="Compute device for Whisper ('cuda', 'cpu', 'auto')",
        show_default=False,
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Spoken language code for Whisper (ISO-639-1) or auto-detect",
        show_default=False,
    ),
    scene_threshold: Optional[float] = typer.Option(
        None,
        "--scene-threshold",
        help="FFmpeg scene detection threshold (0.0 to 1.0)",
        show_default=False,
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "--max-keyframes",
        help="Maximum keyframes to extract from video",
        show_default=False,
    ),
    debounce_seconds: Optional[float] = typer.Option(
        None,
        "--debounce-seconds",
        help="Minimum seconds between extracted keyframes",
        show_default=False,
    ),
    keep_frames: bool = typer.Option(
        True,
        "--keep-frames/--no-keep-frames",
        help="Keep extracted keyframe image files alongside Markdown output",
    ),
    embed_images: bool = typer.Option(
        True,
        "--embed-images/--no-embed-images",
        help="Embed relative keyframe image links inside Markdown document",
    ),
    ollama_url: Optional[str] = typer.Option(
        None,
        "--ollama-url",
        help="Ollama API base URL endpoint",
        show_default=False,
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "--verbose",
        "-v",
        help="Enable verbose debug logging and stack trace output",
    ),
) -> None:
    """Process a single video into publication-grade Markdown documentation."""
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    target_video = video or input
    if target_video is None:
        console.print(
            Panel(
                "[bold red]No input video specified.[/bold red]\n"
                "Provide a video file path as an argument or using [bold]--input / -i[/bold].\n"
                "Run [bold cyan]video2doc process --help[/bold cyan] for full options.",
                title="[bold red]Missing Input[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    target_output = output or output_dir

    overrides = _build_settings_overrides(
        output_path=target_output,
        template=template,
        model_vision=model_vision,
        model_text=model_text,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        language=language,
        scene_threshold=scene_threshold,
        max_frames=max_frames,
        debounce_seconds=debounce_seconds,
        keep_frames=keep_frames,
        embed_images=embed_images,
        ollama_url=ollama_url,
    )

    try:
        settings = get_settings(**overrides)
        pipeline = DocumentPipeline(settings=settings)

        console.print(
            Panel(
                f"[bold cyan]Input Video:[/bold cyan] {escape(str(target_video))}\n"
                f"[bold cyan]Template:[/bold cyan] {escape(str(template))}\n"
                f"[bold cyan]Vision Model:[/bold cyan] {escape(str(settings.vision_model))}\n"
                f"[bold cyan]Text Model:[/bold cyan] {escape(str(settings.text_model))}\n"
                f"[bold cyan]Whisper Model:[/bold cyan] {escape(str(settings.whisper_model))}",
                title="[bold green]Video2Doc Pipeline Started[/bold green]",
                border_style="cyan",
            )
        )

        doc_path = pipeline.process_video(
            video_path=target_video,
            template=template,
            output_dir=target_output,
        )

        console.print(
            Panel(
                f"[bold green]Document generated successfully![/bold green]\n\n"
                f"[bold]Output Markdown:[/bold] {escape(str(doc_path))}",
                title="[bold green]Success[/bold green]",
                border_style="green",
            )
        )
    except Exception as exc:
        _handle_cli_error(exc, debug=debug)


@app.command("batch")
def batch_command(
    input_dir_arg: Optional[Path] = typer.Argument(
        None,
        help="Directory containing video files to process",
        show_default=False,
    ),
    input_dir: Optional[Path] = typer.Option(
        None,
        "--input-dir",
        "-i",
        help="Directory containing video files to process (alternative to argument)",
        show_default=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Target output directory for generated Markdown documents",
        show_default=False,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        help="Target output directory (alias for --output)",
        show_default=False,
    ),
    template: str = typer.Option(
        "sop",
        "--template",
        "-t",
        help="Template name ('sop', 'lecture', 'executive') or path to custom YAML template",
    ),
    model_vision: Optional[str] = typer.Option(
        None,
        "--model-vision",
        "--vision-model",
        help="Ollama vision model name (e.g. 'llama3.2-vision')",
        show_default=False,
    ),
    model_text: Optional[str] = typer.Option(
        None,
        "--model-text",
        "--text-model",
        help="Ollama text model name (e.g. 'qwen2.5-coder:14b')",
        show_default=False,
    ),
    whisper_model: Optional[str] = typer.Option(
        None,
        "--whisper-model",
        help="OpenAI Whisper model variant (tiny, base, small, medium, turbo)",
        show_default=False,
    ),
    whisper_device: Optional[str] = typer.Option(
        None,
        "--whisper-device",
        help="Compute device for Whisper ('cuda', 'cpu', 'auto')",
        show_default=False,
    ),
    language: Optional[str] = typer.Option(
        None,
        "--language",
        "-l",
        help="Spoken language code for Whisper (ISO-639-1) or auto-detect",
        show_default=False,
    ),
    scene_threshold: Optional[float] = typer.Option(
        None,
        "--scene-threshold",
        help="FFmpeg scene detection threshold (0.0 to 1.0)",
        show_default=False,
    ),
    max_frames: Optional[int] = typer.Option(
        None,
        "--max-frames",
        "--max-keyframes",
        help="Maximum keyframes to extract from each video",
        show_default=False,
    ),
    debounce_seconds: Optional[float] = typer.Option(
        None,
        "--debounce-seconds",
        help="Minimum seconds between extracted keyframes",
        show_default=False,
    ),
    keep_frames: bool = typer.Option(
        True,
        "--keep-frames/--no-keep-frames",
        help="Keep extracted keyframe image files alongside Markdown output",
    ),
    embed_images: bool = typer.Option(
        True,
        "--embed-images/--no-embed-images",
        help="Embed relative keyframe image links inside Markdown document",
    ),
    ollama_url: Optional[str] = typer.Option(
        None,
        "--ollama-url",
        help="Ollama API base URL endpoint",
        show_default=False,
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "--verbose",
        "-v",
        help="Enable verbose debug logging and stack trace output",
    ),
) -> None:
    """Batch process an entire directory of instructional videos into Markdown documents."""
    if debug:
        logging.basicConfig(level=logging.DEBUG)

    source_dir = input_dir_arg or input_dir
    if source_dir is None:
        console.print(
            Panel(
                "[bold red]No input directory specified.[/bold red]\n"
                "Provide an input directory as an argument or using [bold]--input-dir / -i[/bold].\n"
                "Run [bold cyan]video2doc batch --help[/bold cyan] for full options.",
                title="[bold red]Missing Input Directory[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    target_output = output or output_dir

    overrides = _build_settings_overrides(
        output_path=target_output,
        template=template,
        model_vision=model_vision,
        model_text=model_text,
        whisper_model=whisper_model,
        whisper_device=whisper_device,
        language=language,
        scene_threshold=scene_threshold,
        max_frames=max_frames,
        debounce_seconds=debounce_seconds,
        keep_frames=keep_frames,
        embed_images=embed_images,
        ollama_url=ollama_url,
    )

    try:
        settings = get_settings(**overrides)
        pipeline = DocumentPipeline(settings=settings)

        console.print(
            Panel(
                f"[bold cyan]Input Directory:[/bold cyan] {escape(str(source_dir))}\n"
                f"[bold cyan]Template:[/bold cyan] {escape(str(template))}\n"
                f"[bold cyan]Output Directory:[/bold cyan] {escape(str(target_output or settings.output_dir))}",
                title="[bold green]Video2Doc Batch Pipeline Started[/bold green]",
                border_style="cyan",
            )
        )

        generated = pipeline.process_batch(
            input_dir=source_dir,
            template=template,
            output_dir=target_output,
        )

        table = Table(title="Batch Processing Summary", border_style="green")
        table.add_column("No.", justify="right", style="cyan", no_wrap=True)
        table.add_column("Generated Document", style="white")
        table.add_column("Status", style="green")

        for idx, doc_path in enumerate(generated, start=1):
            table.add_row(str(idx), escape(str(doc_path.name)), "[green]SUCCESS[/green]")

        console.print(table)
        console.print(
            f"\n[bold green]Batch completed:[/bold green] {len(generated)} document(s) generated successfully."
        )
    except Exception as exc:
        _handle_cli_error(exc, debug=debug)


@templates_app.command("list")
def templates_list_command() -> None:
    """List all available built-in and user-defined documentation templates."""
    engine = TemplateEngine()
    available = engine.list_available_templates()

    table = Table(title="Available Documentation Templates", border_style="cyan")
    table.add_column("Template", style="bold cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Version", style="yellow")
    table.add_column("Description", style="white")

    for tmpl in available:
        table.add_row(
            tmpl.get("name", "unknown"),
            tmpl.get("type", "builtin"),
            tmpl.get("version", "1.0"),
            tmpl.get("description", ""),
        )

    console.print(table)


@templates_app.command("show")
def templates_show_command(
    name: str = typer.Argument(
        ...,
        help="Template name ('sop', 'lecture', 'executive') or path to YAML template",
    ),
) -> None:
    """Inspect detailed schema, prompts, and layout of a specific template."""
    engine = TemplateEngine()
    try:
        template_def = engine.load_template(name)
        sections_str = (
            ", ".join(template_def.sections)
            if template_def.sections
            else "(Default sections)"
        )

        panel_content = (
            f"[bold cyan]Template Name:[/bold cyan] {template_def.name}\n"
            f"[bold cyan]Version:[/bold cyan] {template_def.version}\n"
            f"[bold cyan]Description:[/bold cyan] {template_def.description}\n"
            f"[bold cyan]Sections:[/bold cyan] {sections_str}\n\n"
            f"[bold yellow]Vision Prompt:[/bold yellow]\n{template_def.vision_prompt.strip()}\n\n"
            f"[bold yellow]Synthesis Prompt:[/bold yellow]\n{template_def.synthesis_prompt.strip()}"
        )

        console.print(
            Panel(
                panel_content,
                title=f"Template Details: {template_def.name}",
                border_style="cyan",
            )
        )
    except Exception as exc:
        _handle_cli_error(exc)


@templates_app.command("init")
def templates_init_command(
    output_path: Path = typer.Argument(
        Path("custom_template.yaml"),
        help="Destination path for scaffolding a new custom template YAML",
    ),
) -> None:
    """Scaffold a new custom template YAML with documented fields."""
    starter_content = """# Video2Doc Custom Documentation Template
name: my_custom_template
description: Custom template description for technical documentation
version: "1.0"
sections:
  - Executive Overview
  - System Requirements
  - Implementation Steps
  - Verification & Testing

vision_prompt: >
  Examine the keyframe screenshot carefully.
  Describe active application windows, key configuration parameters,
  visible dialog boxes, code snippets, or diagrams.

synthesis_prompt: >
  # {{ title }}

  ## 1. Executive Overview
  {{ transcription_text }}

  ## 2. Visual Findings & Demonstration
  {{ visual_summary }}

  ## 3. Step-by-Step Instructions
  Reconstruct step-by-step procedures demonstrated in the video.

  ## 4. Verification & Testing
  Provide concrete verification steps to confirm completion.
"""
    try:
        dest = output_path.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(starter_content, encoding="utf-8")
        console.print(
            Panel(
                f"[bold green]Custom template scaffolded successfully![/bold green]\n"
                f"Saved to: [bold]{dest}[/bold]\n\n"
                f"You can use it with:\n"
                f"[bold cyan]video2doc process video.mp4 --template {dest}[/bold cyan]",
                title="Template Scaffolded",
                border_style="green",
            )
        )
    except Exception as exc:
        _handle_cli_error(exc)


@config_app.command("show")
def config_show_command() -> None:
    """Display active Video2Doc configuration resolved across CLI, env, and defaults."""
    settings = get_settings()

    table = Table(title="Active Video2Doc Configuration", border_style="cyan")
    table.add_column("Setting Key", style="bold cyan")
    table.add_column("Resolved Value", style="white")
    table.add_column("Description", style="dim")

    for field_name, field_info in settings.__class__.model_fields.items():
        val = getattr(settings, field_name, "")
        desc = field_info.description or ""
        table.add_row(field_name, str(val), desc)

    console.print(table)


@config_app.command("check")
def config_check_command() -> None:
    """Execute pre-flight environment checks (FFmpeg, ffprobe, Ollama, Models)."""
    settings = get_settings()
    diag = check_environment(settings)

    table = Table(title="Pre-Flight Environment Diagnostics", border_style="cyan")
    table.add_column("Diagnostic Item", style="bold cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details & Remediation", style="white")

    # 1. FFmpeg
    if diag["ffmpeg"]:
        table.add_row("FFmpeg Executable", "[green]OK[/green]", "Found in system PATH")
    else:
        table.add_row(
            "FFmpeg Executable",
            "[red]MISSING[/red]",
            "FFmpeg not found in PATH. Install via winget install ffmpeg or brew install ffmpeg.",
        )

    # 2. ffprobe
    if diag["ffprobe"]:
        table.add_row("ffprobe Executable", "[green]OK[/green]", "Found in system PATH")
    else:
        table.add_row(
            "ffprobe Executable",
            "[red]MISSING[/red]",
            "ffprobe not found in PATH. Verify FFmpeg installation.",
        )

    # 3. Ollama Connectivity
    if diag["ollama"]:
        table.add_row(
            "Ollama Server",
            "[green]ONLINE[/green]",
            f"Connected to {settings.ollama_url}",
        )
    else:
        table.add_row(
            "Ollama Server",
            "[red]UNREACHABLE[/red]",
            f"Cannot connect to {settings.ollama_url}. Is Ollama running? Start with 'ollama serve'. Error: {diag.get('ollama_error')}",
        )

    # 4. Vision Model
    if diag["vision_model"]:
        table.add_row(
            f"Vision Model ({settings.vision_model})",
            "[green]AVAILABLE[/green]",
            "Model pulled and available in Ollama",
        )
    else:
        table.add_row(
            f"Vision Model ({settings.vision_model})",
            "[red]NOT FOUND[/red]",
            f"Pull model with: 'ollama pull {settings.vision_model}'",
        )

    # 5. Text Model
    if diag["text_model"]:
        table.add_row(
            f"Text Model ({settings.text_model})",
            "[green]AVAILABLE[/green]",
            "Model pulled and available in Ollama",
        )
    else:
        table.add_row(
            f"Text Model ({settings.text_model})",
            "[red]NOT FOUND[/red]",
            f"Pull model with: 'ollama pull {settings.text_model}'",
        )

    console.print(table)

    if diag["all_ok"]:
        console.print(
            "\n[bold green]All pre-flight checks passed! System is ready to process videos.[/bold green]"
        )
    else:
        console.print(
            "\n[bold yellow]Some diagnostic checks failed or reported warnings. Check the recommendations above.[/bold yellow]"
        )


if __name__ == "__main__":
    app()
