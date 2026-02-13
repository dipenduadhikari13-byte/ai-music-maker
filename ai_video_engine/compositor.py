"""
Video Compositor — stitch AI-generated clips into a final music video.

Uses ffmpeg (subprocess) for maximum reliability and control over:
  • Crossfade / cut / whip-pan transitions between clips
  • Resolution and frame-rate normalisation
  • Audio muxing
  • Final encoding with professional codec settings
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VideoCompositor:
    """Compose multiple AI-generated video clips into one music video."""

    def __init__(self, output_dir: str | None = None):
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="compositor_")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.ffmpeg = self._find_ffmpeg()

    @staticmethod
    def _find_ffmpeg() -> str:
        """Locate ffmpeg binary."""
        # Try imageio-ffmpeg first (bundled with moviepy)
        try:
            import imageio_ffmpeg
            path = imageio_ffmpeg.get_ffmpeg_exe()
            if path and os.path.isfile(path):
                return path
        except ImportError:
            pass

        # Try system ffmpeg
        path = shutil.which("ffmpeg")
        if path:
            return path

        raise RuntimeError(
            "ffmpeg not found. Install via: sudo apt install ffmpeg  "
            "or: pip install imageio-ffmpeg"
        )

    def _ffprobe(self, path: str) -> Dict:
        """Get video metadata via ffprobe."""
        ffprobe = self.ffmpeg.replace("ffmpeg", "ffprobe")
        if not shutil.which(ffprobe) and not os.path.isfile(ffprobe):
            ffprobe = shutil.which("ffprobe") or "ffprobe"

        try:
            cmd = [
                ffprobe, "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception:
            pass
        return {}

    def get_clip_duration(self, path: str) -> float:
        """Get duration of a video file in seconds."""
        info = self._ffprobe(path)
        try:
            return float(info["format"]["duration"])
        except (KeyError, ValueError, TypeError):
            return 0.0

    # ──────────────────────────────────────────────────────────────────
    #  Normalise a clip to target resolution / fps / codec
    # ──────────────────────────────────────────────────────────────────

    def normalize_clip(
        self,
        input_path: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 24,
    ) -> str:
        """Re-encode a clip to a consistent format for concatenation."""
        out = str(Path(self.output_dir) / f"norm_{uuid.uuid4().hex[:8]}.mp4")
        cmd = [
            self.ffmpeg, "-y", "-i", input_path,
            "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                   f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,"
                   f"fps={fps},format=yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an",  # strip audio — we add the song later
            "-movflags", "+faststart",
            out,
        ]
        self._run(cmd, f"Normalising {Path(input_path).name}")
        return out

    # ──────────────────────────────────────────────────────────────────
    #  Speed-adjust a clip to a target duration
    # ──────────────────────────────────────────────────────────────────

    def adjust_speed(self, input_path: str, target_duration: float) -> str:
        """
        Speed up or slow down a clip so it lasts exactly *target_duration* seconds.
        Uses setpts filter for video.
        """
        actual = self.get_clip_duration(input_path)
        if actual <= 0 or abs(actual - target_duration) < 0.1:
            return input_path

        speed_factor = actual / target_duration  # >1 = slow down, <1 = speed up
        # Clamp to reasonable range
        speed_factor = max(0.25, min(4.0, speed_factor))

        out = str(Path(self.output_dir) / f"speed_{uuid.uuid4().hex[:8]}.mp4")
        pts_expr = f"{1.0 / speed_factor:.4f}*PTS"

        cmd = [
            self.ffmpeg, "-y", "-i", input_path,
            "-vf", f"setpts={pts_expr}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an",
            out,
        ]
        self._run(cmd, f"Speed adjust {speed_factor:.2f}x")
        return out

    # ──────────────────────────────────────────────────────────────────
    #  Looping — extend a short clip to fill a longer scene
    # ──────────────────────────────────────────────────────────────────

    def loop_clip(self, input_path: str, target_duration: float) -> str:
        """
        Loop a clip (forward then reverse for seamless loop) to reach target_duration.
        """
        actual = self.get_clip_duration(input_path)
        if actual <= 0:
            return input_path
        if actual >= target_duration - 0.5:
            return input_path

        loops_needed = int(target_duration / actual) + 1
        out = str(Path(self.output_dir) / f"loop_{uuid.uuid4().hex[:8]}.mp4")

        # Create a concat file with forward + reverse copies
        concat_file = str(Path(self.output_dir) / f"loop_list_{uuid.uuid4().hex[:6]}.txt")
        # Create reversed version
        rev_path = str(Path(self.output_dir) / f"rev_{uuid.uuid4().hex[:8]}.mp4")
        cmd_rev = [
            self.ffmpeg, "-y", "-i", input_path,
            "-vf", "reverse", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            rev_path,
        ]
        self._run(cmd_rev, "Creating reverse for loop")

        # Build concat list: forward, reverse, forward, reverse...
        with open(concat_file, "w") as f:
            for i in range(loops_needed):
                p = input_path if i % 2 == 0 else rev_path
                f.write(f"file '{p}'\n")

        cmd = [
            self.ffmpeg, "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-t", str(target_duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an",
            out,
        ]
        self._run(cmd, f"Looping to {target_duration:.1f}s")
        return out

    # ──────────────────────────────────────────────────────────────────
    #  Stitch clips together
    # ──────────────────────────────────────────────────────────────────

    def stitch_clips(
        self,
        clip_paths: List[str],
        transitions: List[str] | None = None,
        transition_duration: float = 0.5,
        target_resolution: Tuple[int, int] = (1280, 720),
        target_fps: int = 24,
    ) -> str:
        """
        Combine a list of video clips into one file with transitions.

        Parameters
        ----------
        clip_paths : list[str]
            Ordered list of video file paths.
        transitions : list[str] | None
            Transition type between clip[i] and clip[i+1].
            Length must be len(clip_paths) - 1.
            Types: crossfade, cut, fade_black, fade_white, whip_pan, zoom_transition.
        transition_duration : float
            Duration of crossfade transitions in seconds.
        target_resolution : (width, height)
        target_fps : int

        Returns
        -------
        str  — path to stitched video (no audio).
        """
        if not clip_paths:
            raise ValueError("No clips to stitch")

        if len(clip_paths) == 1:
            return self.normalize_clip(
                clip_paths[0], target_resolution[0], target_resolution[1], target_fps
            )

        w, h = target_resolution

        # Normalise all clips first
        normed: List[str] = []
        for p in clip_paths:
            normed.append(self.normalize_clip(p, w, h, target_fps))

        if transitions is None:
            transitions = ["crossfade"] * (len(normed) - 1)
        transitions = transitions[: len(normed) - 1]  # safety

        # Check if all transitions are simple cuts — if so, use concat demuxer
        if all(t == "cut" for t in transitions):
            return self._concat_cut(normed)

        # Otherwise, use xfade filter chain
        return self._concat_xfade(normed, transitions, transition_duration)

    def _concat_cut(self, clips: List[str]) -> str:
        """Simple concatenation with hard cuts (fastest)."""
        out = str(Path(self.output_dir) / f"stitched_{uuid.uuid4().hex[:8]}.mp4")
        concat_file = str(Path(self.output_dir) / f"concat_{uuid.uuid4().hex[:6]}.txt")
        with open(concat_file, "w") as f:
            for p in clips:
                f.write(f"file '{p}'\n")

        cmd = [
            self.ffmpeg, "-y",
            "-f", "concat", "-safe", "0", "-i", concat_file,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-movflags", "+faststart",
            out,
        ]
        self._run(cmd, "Concatenating with hard cuts")
        return out

    def _concat_xfade(
        self,
        clips: List[str],
        transitions: List[str],
        xfade_dur: float,
    ) -> str:
        """
        Concatenate with ffmpeg xfade filter for crossfade / other transitions.
        Builds a sequential xfade filter chain.
        """
        out = str(Path(self.output_dir) / f"stitched_{uuid.uuid4().hex[:8]}.mp4")

        # Get durations of each clip
        durations = []
        for c in clips:
            d = self.get_clip_duration(c)
            durations.append(d if d > 0 else 5.0)

        n = len(clips)

        # Build ffmpeg command with xfade filter chain
        # The xfade filter works pairwise: xfade the first two clips,
        # then xfade the result with the third, etc.

        inputs = []
        for c in clips:
            inputs.extend(["-i", c])

        # Map transition names to ffmpeg xfade transition names
        xfade_map = {
            "crossfade":       "fade",
            "fade_black":      "fadeblack",
            "fade_white":      "fadewhite",
            "whip_pan":        "slideleft",
            "zoom_transition": "radial",
            "cut":             "fade",  # 0-duration = cut
        }

        filter_parts = []
        pad_labels = []
        offset = 0.0

        for i in range(n - 1):
            trans_type = transitions[i] if i < len(transitions) else "crossfade"
            ff_trans = xfade_map.get(trans_type, "fade")
            dur = xfade_dur if trans_type != "cut" else 0.01

            # Compute offset: time in the accumulated output where xfade starts
            if i == 0:
                offset = max(0, durations[0] - dur)
                left = "[0:v]"
                right = "[1:v]"
            else:
                offset = offset + max(0, durations[i] - dur)
                left = f"[v{i}]"
                right = f"[{i + 1}:v]"

            out_label = f"[v{i + 1}]" if i < n - 2 else "[vout]"
            filter_parts.append(
                f"{left}{right}xfade=transition={ff_trans}:duration={dur:.3f}:"
                f"offset={offset:.3f}{out_label}"
            )

        filter_str = ";".join(filter_parts)

        cmd = [
            self.ffmpeg, "-y",
            *inputs,
            "-filter_complex", filter_str,
            "-map", "[vout]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            out,
        ]
        self._run(cmd, "Stitching with transitions")
        return out

    # ──────────────────────────────────────────────────────────────────
    #  Add audio track
    # ──────────────────────────────────────────────────────────────────

    def add_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str | None = None,
        fade_out: float = 2.0,
    ) -> str:
        """Mux the song audio onto the stitched video."""
        out = output_path or str(
            Path(self.output_dir) / f"final_{uuid.uuid4().hex[:8]}.mp4"
        )

        video_dur = self.get_clip_duration(video_path)

        # Audio filter: trim to video duration + fade out at end
        af_parts = [f"atrim=0:{video_dur:.3f}"]
        if fade_out > 0 and video_dur > fade_out:
            af_parts.append(f"afade=t=out:st={video_dur - fade_out:.3f}:d={fade_out:.3f}")

        af = ",".join(af_parts)

        cmd = [
            self.ffmpeg, "-y",
            "-i", video_path,
            "-i", audio_path,
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-af", af,
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k",
            "-shortest",
            "-movflags", "+faststart",
            out,
        ]
        self._run(cmd, "Muxing audio")
        return out

    # ──────────────────────────────────────────────────────────────────
    #  Add intro / outro
    # ──────────────────────────────────────────────────────────────────

    def add_intro_outro(
        self,
        video_path: str,
        intro_text: str = "",
        outro_text: str = "",
        intro_duration: float = 3.0,
        outro_duration: float = 3.0,
    ) -> str:
        """Add optional text-based intro / outro cards."""
        if not intro_text and not outro_text:
            return video_path

        info = self._ffprobe(video_path)
        try:
            streams = info.get("streams", [])
            vs = next(s for s in streams if s["codec_type"] == "video")
            w = int(vs["width"])
            h = int(vs["height"])
            fps = eval(vs.get("r_frame_rate", "24"))  # e.g. "24/1"
        except Exception:
            w, h, fps = 1280, 720, 24

        parts = []

        # Create intro card
        if intro_text:
            intro_path = str(Path(self.output_dir) / f"intro_{uuid.uuid4().hex[:6]}.mp4")
            escaped = intro_text.replace("'", "\\'").replace(":", "\\:")
            cmd = [
                self.ffmpeg, "-y",
                "-f", "lavfi",
                "-i", f"color=c=black:s={w}x{h}:d={intro_duration}:r={fps}",
                "-vf", (
                    f"drawtext=text='{escaped}':fontcolor=white:fontsize=48:"
                    f"x=(w-text_w)/2:y=(h-text_h)/2:alpha='if(lt(t,0.5),t/0.5,1)'"
                ),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                intro_path,
            ]
            self._run(cmd, "Creating intro card")
            parts.append(intro_path)

        parts.append(video_path)

        # Create outro card
        if outro_text:
            outro_path = str(Path(self.output_dir) / f"outro_{uuid.uuid4().hex[:6]}.mp4")
            escaped = outro_text.replace("'", "\\'").replace(":", "\\:")
            cmd = [
                self.ffmpeg, "-y",
                "-f", "lavfi",
                "-i", f"color=c=black:s={w}x{h}:d={outro_duration}:r={fps}",
                "-vf", (
                    f"drawtext=text='{escaped}':fontcolor=white:fontsize=48:"
                    f"x=(w-text_w)/2:y=(h-text_h)/2:"
                    f"alpha='if(gt(t,{outro_duration - 0.5}),(({outro_duration}-t)/0.5),1)'"
                ),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                outro_path,
            ]
            self._run(cmd, "Creating outro card")
            parts.append(outro_path)

        if len(parts) == 1:
            return parts[0]

        return self._concat_cut(parts)

    # ──────────────────────────────────────────────────────────────────
    #  Utility
    # ──────────────────────────────────────────────────────────────────

    def _run(self, cmd: List[str], desc: str = "") -> subprocess.CompletedProcess:
        logger.info("ffmpeg: %s", desc)
        logger.debug("CMD: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            logger.error("ffmpeg STDERR:\n%s", result.stderr[-2000:])
            raise RuntimeError(f"ffmpeg failed ({desc}): {result.stderr[-500:]}")
        return result

    def cleanup(self):
        """Remove temporary files."""
        import shutil as _shutil
        try:
            _shutil.rmtree(self.output_dir, ignore_errors=True)
        except Exception:
            pass
