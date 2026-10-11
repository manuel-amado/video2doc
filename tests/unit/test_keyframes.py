"""Unit tests for Video2Doc keyframe extraction pipeline."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video2doc.config import Settings
from video2doc.exceptions import MediaCorruptError, MissingDependencyError
from video2doc.vision.keyframes import Keyframe, KeyframeExtractor


# ==============================================================================
# Data Model Tests
# ==============================================================================

def test_keyframe_dataclass_fields():
    """Keyframe dataclass correctly holds timestamp, frame_path, and score."""
    p = Path("frames/frame_0001.jpg")
    kf = Keyframe(timestamp=12.5, frame_path=p, score=0.35)
    assert kf.timestamp == 12.5
    assert kf.frame_path == p
    assert kf.score == 0.35


def test_keyframe_default_score():
    """Keyframe score defaults to 0.0."""
    p = Path("frames/frame_0002.jpg")
    kf = Keyframe(timestamp=5.0, frame_path=p)
    assert kf.score == 0.0


# ==============================================================================
# Initialization & Configuration Tests
# ==============================================================================

def test_keyframe_extractor_defaults():
    """KeyframeExtractor loads default settings from application configuration."""
    extractor = KeyframeExtractor()
    assert extractor.threshold == 0.3
    assert extractor.debounce_seconds == 3.0
    assert extractor.max_frames == 10


def test_keyframe_extractor_custom_init():
    """KeyframeExtractor accepts explicit constructor parameters."""
    extractor = KeyframeExtractor(threshold=0.45, debounce_seconds=5.0, max_frames=20)
    assert extractor.threshold == 0.45
    assert extractor.debounce_seconds == 5.0
    assert extractor.max_frames == 20


def test_keyframe_extractor_settings_override():
    """KeyframeExtractor accepts custom Settings instance."""
    settings = Settings(scene_threshold=0.5, debounce_seconds=4.0, max_keyframes=15)
    extractor = KeyframeExtractor(settings=settings)
    assert extractor.threshold == 0.5
    assert extractor.debounce_seconds == 4.0
    assert extractor.max_frames == 15


# ==============================================================================
# Duration Probing Tests
# ==============================================================================

def test_probe_duration_format_entry(sample_video: Path, ffmpeg_state):
    """Probe duration extracts duration via format=duration."""
    ffmpeg_state.mock_duration = 120.5
    extractor = KeyframeExtractor()
    dur = extractor._probe_duration(sample_video)
    assert dur == 120.5


def test_probe_duration_stream_fallback(sample_video: Path):
    """Probe duration falls back to stream=duration when format=duration is N/A."""
    calls: list[list[str]] = []

    def fake_check_output(cmd, *args, **kwargs):
        cmd_list = [str(c) for c in cmd]
        calls.append(cmd_list)
        if "format=duration" in " ".join(cmd_list):
            return b"N/A\n"
        if "stream=duration" in " ".join(cmd_list):
            return b"45.2\n"
        return b"60.0\n"

    with patch("subprocess.check_output", side_effect=fake_check_output):
        extractor = KeyframeExtractor()
        dur = extractor._probe_duration(sample_video)
        assert dur == 45.2
        assert any("stream=duration" in " ".join(c) for c in calls)


def test_probe_duration_corrupt_file_raises_media_corrupt(sample_video: Path, ffmpeg_state):
    """Failed duration probing raises MediaCorruptError."""
    ffmpeg_state.simulate_corrupt_video = True
    extractor = KeyframeExtractor()
    with pytest.raises(MediaCorruptError):
        extractor._probe_duration(sample_video)


# ==============================================================================
# Dynamic Scene Detection & Interval Fallback Tests
# ==============================================================================

def test_extract_dynamic_scene_detection_happy_path(sample_video: Path, tmp_path: Path):
    """Dynamic scene detection parses frames, timestamps, and respects max_frames."""
    out_dir = tmp_path / "frames"
    extractor = KeyframeExtractor(threshold=0.35, debounce_seconds=2.5, max_frames=2)

    # Simulate dynamic FFmpeg run producing 3 frames with showinfo output in stderr
    fake_frames = [out_dir / f"frame_{i:04d}.jpg" for i in range(1, 4)]

    def fake_run(cmd, *args, **kwargs):
        cmd_list = [str(c) for c in cmd]
        out_dir.mkdir(parents=True, exist_ok=True)
        # Create digit-named frames
        for f in fake_frames:
            f.write_bytes(b"\xff\xd8\xff\xd9")

        res = MagicMock(spec=subprocess.CompletedProcess)
        res.returncode = 0
        res.stderr = (
            b"[Parsed_showinfo_1 @ 0x1] n:0 pts_time:5.200000 pos:100\n"
            b"[Parsed_showinfo_1 @ 0x1] n:1 pts_time:18.400000 pos:200\n"
            b"[Parsed_showinfo_1 @ 0x1] n:2 pts_time:35.600000 pos:300\n"
        )
        return res

    with patch("subprocess.run", side_effect=fake_run), patch.object(
        extractor, "_probe_duration", return_value=60.0
    ):
        keyframes = extractor.extract(sample_video, out_dir)

    # 3 frames were produced, but max_frames was 2, so it bounded to 2
    assert len(keyframes) == 2
    assert keyframes[0].timestamp == 5.2
    assert keyframes[1].timestamp in (18.4, 35.6)
    assert all(kf.frame_path.exists() for kf in keyframes)


def test_extract_fallback_interval_sampling_when_under_two_frames(
    sample_video: Path, tmp_path: Path, ffmpeg_state
):
    """When dynamic detection yields fewer than 2 frames, fallback interval sampling is triggered."""
    out_dir = tmp_path / "frames"
    extractor = KeyframeExtractor()
    keyframes = extractor.extract(sample_video, out_dir, max_frames=3)

    assert len(keyframes) == 3
    assert all(kf.frame_path.exists() for kf in keyframes)
    # Timestamps spaced along duration
    assert keyframes[0].timestamp < keyframes[1].timestamp < keyframes[2].timestamp
    # Verify dynamic scene detection was executed followed by interval sampling
    executed = " ".join(" ".join(c) for c in ffmpeg_state.calls)
    assert "scene" in executed
    assert "-ss" in executed


def test_extract_bounds_to_max_frames_single(sample_video: Path, tmp_path: Path, ffmpeg_state):
    """Keyframe extraction with max_frames=1 yields exactly 1 keyframe."""
    out_dir = tmp_path / "frames"
    extractor = KeyframeExtractor()
    keyframes = extractor.extract(sample_video, out_dir, max_frames=1)

    assert len(keyframes) == 1
    assert keyframes[0].frame_path.exists()


def test_extract_subprocess_invoked_as_list_without_shell(sample_video: Path, tmp_path: Path):
    """Subprocess invocations pass arguments as a list and never use shell=True."""
    out_dir = tmp_path / "frames"
    extractor = KeyframeExtractor()

    executed_cmds = []

    def spy_run(cmd, *args, **kwargs):
        assert isinstance(cmd, list)
        assert kwargs.get("shell") is not True
        executed_cmds.append(cmd)
        # Create output file if requested
        if str(cmd[-1]).endswith(".jpg"):
            p = Path(cmd[-1])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"\xff\xd8\xff\xd9")
        res = MagicMock(spec=subprocess.CompletedProcess)
        res.returncode = 0
        res.stderr = b""
        return res

    with patch("subprocess.run", side_effect=spy_run), patch.object(
        extractor, "_probe_duration", return_value=60.0
    ):
        extractor.extract(sample_video, out_dir, max_frames=2)

    assert len(executed_cmds) > 0
    # First command is dynamic scene detection
    assert executed_cmds[0][0] == "ffmpeg"
    assert any("gt(scene," in arg for arg in executed_cmds[0])
    assert any("flags=lanczos" in arg for arg in executed_cmds[0])


# ==============================================================================
# Error Handling Tests
# ==============================================================================

def test_extract_missing_video_raises_media_corrupt(tmp_path: Path):
    """Non-existent video file raises MediaCorruptError."""
    missing = tmp_path / "non_existent.mp4"
    extractor = KeyframeExtractor()
    with pytest.raises(MediaCorruptError) as exc_info:
        extractor.extract(missing, tmp_path / "frames")
    assert "does not exist" in exc_info.value.reason.lower()


def test_extract_empty_video_raises_media_corrupt(tmp_path: Path):
    """Zero-byte video file raises MediaCorruptError."""
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    extractor = KeyframeExtractor()
    with pytest.raises(MediaCorruptError) as exc_info:
        extractor.extract(empty, tmp_path / "frames")
    assert "0 bytes" in exc_info.value.reason.lower() or "empty" in exc_info.value.reason.lower()


def test_extract_missing_ffmpeg_raises_missing_dependency(sample_video: Path, ffmpeg_state, tmp_path: Path):
    """Missing FFmpeg executable raises MissingDependencyError."""
    ffmpeg_state.simulate_missing = True
    extractor = KeyframeExtractor()
    with pytest.raises(MissingDependencyError) as exc_info:
        extractor.extract(sample_video, tmp_path / "frames")
    assert exc_info.value.dependency_name == "ffmpeg"


def test_extract_corrupt_video_during_dynamic_detection(sample_video: Path, tmp_path: Path):
    """FFmpeg error during dynamic scene detection raises MediaCorruptError."""
    extractor = KeyframeExtractor()
    with patch.object(extractor, "_probe_duration", return_value=60.0):
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"Invalid video data")):
            with pytest.raises(MediaCorruptError) as exc_info:
                extractor.extract(sample_video, tmp_path / "frames")
            assert "FFmpeg dynamic scene extraction failed" in exc_info.value.reason


def test_extract_corrupt_video_during_fallback_sampling(sample_video: Path, tmp_path: Path):
    """FFmpeg error during fallback interval extraction raises MediaCorruptError."""
    extractor = KeyframeExtractor()
    call_count = 0

    def fail_on_fallback(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Dynamic scene detection passes but produces 0 frames
            res = MagicMock(spec=subprocess.CompletedProcess)
            res.returncode = 0
            res.stderr = b""
            return res
        raise subprocess.CalledProcessError(1, cmd, stderr=b"Decoder error")

    with patch.object(extractor, "_probe_duration", return_value=60.0):
        with patch("subprocess.run", side_effect=fail_on_fallback):
            with pytest.raises(MediaCorruptError) as exc_info:
                extractor.extract(sample_video, tmp_path / "frames")
            assert "FFmpeg fallback interval extraction failed" in exc_info.value.reason
