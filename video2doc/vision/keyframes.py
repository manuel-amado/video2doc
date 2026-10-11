"""Smart keyframe extraction using FFmpeg scene change detection and visual sampling."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video2doc.config import Settings, get_settings
from video2doc.exceptions import MediaCorruptError, MissingDependencyError

logger = logging.getLogger(__name__)


@dataclass
class Keyframe:
    """Represents an extracted visual keyframe with timestamp and file location."""

    timestamp: float
    frame_path: Path
    score: float = 0.0


class KeyframeExtractor:
    """Extracts significant visual keyframes from video files using FFmpeg filtergraphs."""

    def __init__(
        self,
        settings: Settings | None = None,
        threshold: float | None = None,
        debounce_seconds: float | None = None,
        max_frames: int | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.threshold = threshold if threshold is not None else self.settings.scene_threshold
        self.debounce_seconds = (
            debounce_seconds if debounce_seconds is not None else self.settings.debounce_seconds
        )
        self.max_frames = max_frames if max_frames is not None else self.settings.max_keyframes

    def _probe_duration(self, video_path: Path) -> float:
        """Probe total duration of the video file in seconds using ffprobe."""
        ffprobe_bin = shutil.which("ffprobe")
        if not ffprobe_bin:
            if not shutil.which("ffmpeg"):
                raise MissingDependencyError(
                    "ffmpeg",
                    message="FFmpeg executable was not found in system PATH.",
                    remediation="Please install FFmpeg and ensure it is available in system PATH.",
                )
            return 60.0

        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            out_bytes = subprocess.check_output(cmd)
            out_str = out_bytes.decode("utf-8", errors="replace").strip()
            if out_str and out_str != "N/A":
                return max(0.1, float(out_str))
        except (subprocess.CalledProcessError, subprocess.SubprocessError) as err:
            err_msg = (
                err.stderr.decode("utf-8", errors="replace").strip()
                if hasattr(err, "stderr") and isinstance(err.stderr, bytes)
                else str(err)
            )
            raise MediaCorruptError(
                str(video_path),
                reason=f"ffprobe failed to read video metadata: {err_msg}",
            ) from err
        except ValueError:
            pass

        stream_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            out_bytes = subprocess.check_output(stream_cmd)
            out_str = out_bytes.decode("utf-8", errors="replace").strip()
            if out_str and out_str != "N/A":
                return max(0.1, float(out_str))
        except (subprocess.CalledProcessError, subprocess.SubprocessError) as err:
            raise MediaCorruptError(
                str(video_path),
                reason=f"ffprobe stream probe failed: {err}",
            ) from err
        except ValueError:
            pass

        return 60.0

    def extract(
        self,
        video_path: Path | str,
        output_dir: Path | str,
        max_frames: int | None = None,
        threshold: float | None = None,
        debounce_seconds: float | None = None,
    ) -> list[Keyframe]:
        """Extract keyframes using dynamic scene detection with fallback interval sampling.

        Args:
            video_path: Path to the input video file.
            output_dir: Directory where keyframe image files will be saved.
            max_frames: Maximum number of frames to return (defaults to configured max).
            threshold: Scene change sensitivity threshold between 0.0 and 1.0.
            debounce_seconds: Minimum seconds elapsed between consecutive keyframes.

        Returns:
            List of Keyframe objects ordered chronologically.
        """
        video = Path(video_path)
        if not video.exists():
            raise MediaCorruptError(str(video), reason="Video file does not exist.")
        if video.stat().st_size == 0:
            raise MediaCorruptError(str(video), reason="Video file is empty (0 bytes).")

        if not shutil.which("ffmpeg"):
            raise MissingDependencyError(
                "ffmpeg",
                message="FFmpeg executable was not found in system PATH.",
                remediation="Please install FFmpeg and ensure it is available in system PATH.",
            )

        target_max = max_frames if max_frames is not None else self.max_frames
        target_thresh = threshold if threshold is not None else self.threshold
        target_debounce = (
            debounce_seconds if debounce_seconds is not None else self.debounce_seconds
        )

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        duration = self._probe_duration(video)

        # 1. Dynamic scene detection via FFmpeg filtergraph
        filtergraph = (
            f"select='and(gt(scene,{target_thresh}),gte(t-prev_selected_t,{target_debounce}))',"
            f"scale='min(1280,iw)':-1:flags=lanczos"
        )
        pattern = out_path / "frame_%04d.jpg"

        dynamic_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video),
            "-vf",
            filtergraph,
            "-vsync",
            "vfr",
            "-q:v",
            "2",
            str(pattern),
        ]

        try:
            proc = subprocess.run(
                dynamic_cmd,
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.SubprocessError) as err:
            err_msg = (
                err.stderr.decode("utf-8", errors="replace").strip()
                if hasattr(err, "stderr") and isinstance(err.stderr, bytes)
                else str(err)
            )
            raise MediaCorruptError(
                str(video),
                reason=f"FFmpeg dynamic scene extraction failed: {err_msg}",
            ) from err

        mock_literal_file = out_path / "frame_%04d.jpg"
        if mock_literal_file.exists():
            try:
                mock_literal_file.unlink()
            except OSError:
                pass

        digit_frames = sorted(
            [f for f in out_path.glob("frame_[0-9][0-9][0-9][0-9].jpg") if f.is_file()]
        )

        keyframes: list[Keyframe] = []

        if len(digit_frames) < 2:
            logger.info(
                "Dynamic scene detection found %d frame(s) (< 2). Executing fallback interval sampling.",
                len(digit_frames),
            )
            for f in digit_frames:
                try:
                    f.unlink()
                except OSError:
                    pass

            keyframes = self._fallback_interval_sampling(video, out_path, duration, target_max)
        else:
            stderr_text = (
                proc.stderr.decode("utf-8", errors="replace")
                if isinstance(proc.stderr, bytes)
                else str(proc.stderr)
            )
            pts_times = [float(p) for p in re.findall(r"pts_time:([0-9.]+)", stderr_text)]

            timestamps: list[float] = []
            for idx in range(len(digit_frames)):
                if idx < len(pts_times):
                    timestamps.append(pts_times[idx])
                else:
                    timestamps.append(round(duration * (idx + 1) / (len(digit_frames) + 1), 3))

            if len(digit_frames) > target_max:
                step = len(digit_frames) / target_max
                chosen_indices = set(int(i * step) for i in range(target_max))
                for idx, (f_path, ts) in enumerate(zip(digit_frames, timestamps)):
                    if idx in chosen_indices:
                        keyframes.append(
                            Keyframe(timestamp=ts, frame_path=f_path, score=target_thresh)
                        )
                    else:
                        try:
                            f_path.unlink()
                        except OSError:
                            pass
            else:
                for f_path, ts in zip(digit_frames, timestamps):
                    keyframes.append(
                        Keyframe(timestamp=ts, frame_path=f_path, score=target_thresh)
                    )

        if len(keyframes) > target_max:
            excess = keyframes[target_max:]
            keyframes = keyframes[:target_max]
            for extra in excess:
                if extra.frame_path.exists():
                    try:
                        extra.frame_path.unlink()
                    except OSError:
                        pass

        return keyframes

    def _fallback_interval_sampling(
        self,
        video: Path,
        out_dir: Path,
        duration: float,
        max_frames: int,
    ) -> list[Keyframe]:
        """Extract keyframes at regular intervals when dynamic scene change yields insufficient frames."""
        sample_count = max(1, min(max_frames, 3)) if max_frames >= 1 else 1

        keyframes: list[Keyframe] = []
        safe_duration = max(0.1, duration)

        for i in range(1, sample_count + 1):
            ts = safe_duration * (i / (sample_count + 1))
            out_file = out_dir / f"frame_{i:04d}.jpg"
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{ts:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                "scale='min(1280,iw)':-1:flags=lanczos",
                "-q:v",
                "2",
                str(out_file),
            ]
            try:
                subprocess.run(
                    cmd,
                    capture_output=True,
                    check=True,
                )
            except (subprocess.CalledProcessError, subprocess.SubprocessError) as err:
                err_msg = (
                    err.stderr.decode("utf-8", errors="replace").strip()
                    if hasattr(err, "stderr") and isinstance(err.stderr, bytes)
                    else str(err)
                )
                raise MediaCorruptError(
                    str(video),
                    reason=f"FFmpeg fallback interval extraction failed: {err_msg}",
                ) from err

            keyframes.append(Keyframe(timestamp=round(ts, 3), frame_path=out_file, score=0.0))

        return keyframes
