"""
Music Video Pipeline — end-to-end orchestration.

    Audio file  ──► Audio Analysis  ──► Scene Planning (Gemini)
                                            │
    Lyrics (opt) ──────────────────────────►│
    Ref images (opt) ──────────────────────►│
                                            ▼
                                    Scene Plan (N scenes)
                                            │
                          ┌─────────────────┤
                          ▼                 ▼  … (parallel)
                    AI Video Gen      AI Video Gen
                      clip 1            clip N
                          │                 │
                          ▼                 ▼
                    Normalise + Speed Adjust
                          │                 │
                          └────────┬────────┘
                                   ▼
                          Stitch with Transitions
                                   │
                                   ▼
                            Mux Song Audio
                                   │
                                   ▼
                        ★ Final Music Video ★
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .audio_analyzer import AudioAnalyzer, AudioAnalysis
from .compositor import VideoCompositor
from .generators import (
    BaseVideoGenerator,
    GeneratedClip,
    GenerationConfig,
    VideoGeneratorFactory,
)
from .scene_planner import ScenePlanner, VideoScene

logger = logging.getLogger(__name__)


class PipelineProgress:
    """Thread-safe progress tracker."""

    def __init__(self, callback: Callable[[str, float], None] | None = None):
        self._cb = callback
        self.stage = ""
        self.pct = 0.0
        self.detail = ""

    def update(self, stage: str, pct: float, detail: str = ""):
        self.stage = stage
        self.pct = pct
        self.detail = detail
        if self._cb:
            self._cb(f"[{stage}] {detail}", pct)


class MusicVideoPipeline:
    """
    Full music-video generation pipeline.

    Usage
    -----
        pipeline = MusicVideoPipeline(
            video_backend="replicate",
            video_api_key="r8_...",
            video_model="wan2.1-720p",
            gemini_api_key="AI...",
        )
        result = pipeline.generate(
            audio_path="song.mp3",
            lyrics="...",
            style="cinematic",
        )
        print(result)   # path to the final .mp4
    """

    def __init__(
        self,
        video_backend: str,
        video_api_key: str,
        video_model: str | None = None,
        gemini_api_key: str | None = None,
        output_dir: str | None = None,
        max_parallel: int = 2,
    ):
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="mvpipeline_")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Sub-component directories
        clips_dir = str(Path(self.output_dir) / "clips")
        comp_dir = str(Path(self.output_dir) / "comp")

        self.generator: BaseVideoGenerator = VideoGeneratorFactory.create(
            backend=video_backend,
            api_key=video_api_key,
            model=video_model,
            output_dir=clips_dir,
        )
        self.scene_planner: ScenePlanner | None = None
        if gemini_api_key:
            self.scene_planner = ScenePlanner(gemini_api_key)

        self.audio_analyzer = AudioAnalyzer()
        self.compositor = VideoCompositor(output_dir=comp_dir)
        self.max_parallel = max(1, min(max_parallel, 4))
        self.last_scenes: list = []       # scenes used in last generate() call
        self.last_audio_info = None       # AudioAnalysis from last generate()

    # ══════════════════════════════════════════════════════════════════
    #  Main Entry Point
    # ══════════════════════════════════════════════════════════════════

    def generate(
        self,
        audio_path: str,
        lyrics: str = "",
        style: str = "cinematic",
        resolution: Tuple[int, int] = (1280, 720),
        fps: int = 24,
        num_scenes: int | None = None,
        reference_images: List[str] | None = None,
        custom_instructions: str = "",
        clip_duration_hint: float = 5.0,
        intro_text: str = "",
        outro_text: str = "",
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> str:
        """
        Generate a complete music video.

        Parameters
        ----------
        audio_path : str
            Path to the song (mp3/wav).
        lyrics : str
            Song lyrics (optional but recommended).
        style : str
            Visual style preset or custom description.
        resolution : (width, height)
        fps : int
        num_scenes : int | None
            Number of scenes. Auto-calculated if None.
        reference_images : list[str] | None
            Paths to reference images for I2V generation.
        custom_instructions : str
            Extra creative direction.
        clip_duration_hint : float
            Approximate duration of each generated clip (model-dependent).
        intro_text : str
            Text to show on intro card.
        outro_text : str
            Text to show on outro card.
        progress_callback : callable | None
            Called with (message, progress_0_to_1).

        Returns
        -------
        str — path to the final .mp4 music video.
        """
        progress = PipelineProgress(progress_callback)
        total_t0 = time.time()

        # ── 1. Audio Analysis ──────────────────────────────────────
        progress.update("Audio Analysis", 0.05, "Analysing song structure…")
        audio_info = self.audio_analyzer.analyze(audio_path)
        logger.info(
            "Audio: %.1fs  %s BPM  mood=%s  key=%s  sections=%d",
            audio_info.duration, audio_info.tempo, audio_info.mood,
            audio_info.key, len(audio_info.sections),
        )

        # ── 2. Scene Planning ──────────────────────────────────────
        progress.update("Scene Planning", 0.10, "Creating cinematic scene plan…")

        if self.scene_planner:
            ref_descs = None
            if reference_images:
                ref_descs = [f"User-provided image: {Path(p).name}" for p in reference_images]

            scenes = self.scene_planner.plan_scenes(
                audio_analysis=audio_info,
                lyrics=lyrics,
                style=style,
                num_scenes=num_scenes,
                custom_instructions=custom_instructions,
                reference_image_descriptions=ref_descs,
            )
        else:
            scenes = self._simple_scene_plan(audio_info, style, num_scenes)

        logger.info("Scene plan: %d scenes", len(scenes))
        self.last_scenes = scenes        # expose for UI display
        self.last_audio_info = audio_info
        for s in scenes:
            logger.info(
                "  Scene %d: %.1f–%.1fs  (%s) %s",
                s.scene_id, s.start_time, s.end_time, s.mood, s.visual_prompt[:80],
            )

        # ── 3. Video Generation ────────────────────────────────────
        progress.update("Video Generation", 0.15, f"Generating {len(scenes)} video clips…")
        clips = self._generate_clips(
            scenes=scenes,
            reference_images=reference_images,
            config=GenerationConfig(
                width=resolution[0],
                height=resolution[1],
                fps=fps,
            ),
            clip_duration_hint=clip_duration_hint,
            progress=progress,
        )

        if not clips:
            raise RuntimeError("No video clips were generated. Check API keys and try again.")

        logger.info("Generated %d clips", len(clips))

        # ── 4. Prepare clips (normalise, speed-adjust, loop) ──────
        progress.update("Processing", 0.75, "Normalising and time-stretching clips…")
        prepared_paths = self._prepare_clips(clips, scenes, resolution, fps)

        # ── 5. Stitch with transitions ────────────────────────────
        progress.update("Composition", 0.85, "Stitching clips with transitions…")
        transitions = [s.transition for s in scenes[:-1]] if len(scenes) > 1 else []
        stitched = self.compositor.stitch_clips(
            clip_paths=prepared_paths,
            transitions=transitions,
            transition_duration=0.5,
            target_resolution=resolution,
            target_fps=fps,
        )

        # ── 5b. Optional intro/outro ──────────────────────────────
        if intro_text or outro_text:
            progress.update("Composition", 0.88, "Adding intro/outro…")
            stitched = self.compositor.add_intro_outro(
                stitched, intro_text, outro_text,
            )

        # ── 6. Mux audio ─────────────────────────────────────────
        progress.update("Audio Sync", 0.92, "Synchronising song audio…")
        final_path = str(Path(self.output_dir) / "music_video_final.mp4")
        final_path = self.compositor.add_audio(
            video_path=stitched,
            audio_path=audio_path,
            output_path=final_path,
            fade_out=2.0,
        )

        total_time = time.time() - total_t0
        size_mb = os.path.getsize(final_path) / 1_048_576
        progress.update("Done", 1.0, f"Complete! {size_mb:.0f} MB in {total_time:.0f}s")

        logger.info(
            "Music Video Complete: %s  (%.1f MB, %.0fs generation time)",
            final_path, size_mb, total_time,
        )

        return final_path

    # ══════════════════════════════════════════════════════════════════
    #  Clip Generation (with parallelism)
    # ══════════════════════════════════════════════════════════════════

    def _generate_clips(
        self,
        scenes: List[VideoScene],
        reference_images: List[str] | None,
        config: GenerationConfig,
        clip_duration_hint: float,
        progress: PipelineProgress,
    ) -> List[GeneratedClip]:
        """Generate video clips for all scenes, optionally in parallel."""
        clips: List[Optional[GeneratedClip]] = [None] * len(scenes)
        total = len(scenes)

        def _gen_one(idx: int, scene: VideoScene) -> Tuple[int, GeneratedClip | None]:
            """Generate a single clip (runs in thread)."""
            try:
                logger.info("Generating clip %d/%d: %s", idx + 1, total, scene.visual_prompt[:60])

                def _cb(msg):
                    progress.update(
                        "Video Generation",
                        0.15 + 0.55 * (idx / total),
                        f"Scene {idx + 1}/{total}: {msg}",
                    )

                # Decide: image-to-video or text-to-video
                ref_img = None
                if reference_images and idx < len(reference_images):
                    ref_img = reference_images[idx]
                elif reference_images and reference_images:
                    # Cycle through reference images
                    ref_img = reference_images[idx % len(reference_images)]

                if ref_img and os.path.isfile(ref_img):
                    clip = self.generator.image_to_video(
                        image_path=ref_img,
                        prompt=scene.visual_prompt,
                        config=config,
                        progress_cb=_cb,
                    )
                else:
                    clip = self.generator.text_to_video(
                        prompt=scene.visual_prompt,
                        config=config,
                        progress_cb=_cb,
                    )

                clip.scene_id = scene.scene_id
                return idx, clip

            except Exception as exc:
                logger.error("Failed to generate scene %d: %s", idx + 1, exc)
                return idx, None

        # Generate clips (parallel or sequential)
        if self.max_parallel > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                futures = {
                    pool.submit(_gen_one, i, s): i
                    for i, s in enumerate(scenes)
                }
                for future in concurrent.futures.as_completed(futures):
                    idx, clip = future.result()
                    if clip:
                        clips[idx] = clip
                    progress.update(
                        "Video Generation",
                        0.15 + 0.55 * (sum(1 for c in clips if c is not None) / total),
                        f"Completed {sum(1 for c in clips if c is not None)}/{total} clips",
                    )
        else:
            for i, scene in enumerate(scenes):
                idx, clip = _gen_one(i, scene)
                if clip:
                    clips[idx] = clip
                progress.update(
                    "Video Generation",
                    0.15 + 0.55 * ((i + 1) / total),
                    f"Completed {i + 1}/{total} clips",
                )

        # Filter out failures
        result = [c for c in clips if c is not None]
        if len(result) < len(scenes):
            logger.warning(
                "%d/%d clips generated (some failed)", len(result), len(scenes)
            )
        return result

    # ══════════════════════════════════════════════════════════════════
    #  Clip Preparation (normalise + time adjust + loop)
    # ══════════════════════════════════════════════════════════════════

    def _prepare_clips(
        self,
        clips: List[GeneratedClip],
        scenes: List[VideoScene],
        resolution: Tuple[int, int],
        fps: int,
    ) -> List[str]:
        """
        For each clip:
        1. Normalise to target resolution / fps / codec.
        2. If the clip is shorter than the scene, loop it.
        3. If the clip is much longer, trim/speed-adjust.
        """
        prepared: List[str] = []
        w, h = resolution

        for i, clip in enumerate(clips):
            scene = scenes[i] if i < len(scenes) else scenes[-1]
            target_dur = scene.duration

            # 1. Normalise
            normed = self.compositor.normalize_clip(clip.path, w, h, fps)

            # 2. Get actual duration
            actual_dur = self.compositor.get_clip_duration(normed)
            if actual_dur <= 0:
                actual_dur = clip.duration

            # 3. Adjust timing
            if actual_dur < target_dur - 1.0:
                # Clip is too short → loop it
                normed = self.compositor.loop_clip(normed, target_dur)
            elif actual_dur > target_dur + 2.0:
                # Clip is too long → speed adjust (mild) or trim
                if actual_dur / target_dur > 2.0:
                    # Too much speedup needed, just trim
                    normed = self._trim_clip(normed, target_dur)
                else:
                    normed = self.compositor.adjust_speed(normed, target_dur)

            prepared.append(normed)

        return prepared

    def _trim_clip(self, path: str, duration: float) -> str:
        """Trim a clip to a specific duration."""
        out = str(Path(self.output_dir) / f"trim_{id(path)}.mp4")
        cmd = [
            self.compositor.ffmpeg, "-y",
            "-i", path,
            "-t", str(duration),
            "-c:v", "copy", "-an",
            out,
        ]
        self.compositor._run(cmd, f"Trimming to {duration:.1f}s")
        return out

    # ══════════════════════════════════════════════════════════════════
    #  Fallback scene plan (if no Gemini key)
    # ══════════════════════════════════════════════════════════════════

    def _simple_scene_plan(
        self,
        audio_info: AudioAnalysis,
        style: str,
        num_scenes: int | None,
    ) -> List[VideoScene]:
        """Create a basic scene plan without LLM assistance."""
        n = num_scenes or max(4, int(audio_info.duration / 15))
        step = audio_info.duration / n

        generic_prompts = [
            f"Cinematic sweeping shot of a beautiful landscape, {style} style, dramatic lighting, camera slowly moving forward",
            f"Close-up artistic shot with dynamic motion and depth, {style} style, vivid colors, particles in the air",
            f"Wide establishing shot of an atmospheric environment, {style} style, volumetric lighting, epic scale",
            f"Intimate close-up with shallow depth of field, {style} style, soft bokeh, emotional and expressive",
            f"Dynamic tracking shot through a visually stunning scene, {style} style, fluid camera movement",
            f"Aerial view slowly descending over a breathtaking vista, {style} style, golden hour lighting",
        ]

        scenes = []
        for i in range(n):
            start = round(i * step, 2)
            end = round(min((i + 1) * step, audio_info.duration), 2)
            prompt = generic_prompts[i % len(generic_prompts)]

            scenes.append(VideoScene(
                scene_id=i + 1,
                start_time=start,
                end_time=end,
                duration=round(end - start, 2),
                visual_prompt=prompt,
                negative_prompt="blurry, low quality, text, watermark, static, still image",
                camera_motion="slow dolly forward",
                style_notes=style,
                transition="crossfade" if i < n - 1 else "fade_black",
                mood=audio_info.mood,
            ))

        return scenes

    # ══════════════════════════════════════════════════════════════════
    #  Cost Estimation
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def estimate_cost(
        duration: float,
        backend: str,
        model: str = "",
        clip_duration: float = 5.0,
    ) -> Dict[str, float | int | str]:
        """
        Estimate generation time and cost before starting.

        Returns dict with: num_clips, est_cost_usd, est_time_minutes.
        """
        num_clips = max(1, int(duration / clip_duration))

        cost_map = {
            "replicate": {
                "wan2.1-480p": 0.15, "wan2.1-i2v-fast": 0.10,
                "cogvideox": 0.25, "animatediff": 0.08,
                "kling-v2.1": 0.35, "luma-ray": 0.30,
                "luma-flash": 0.06, "minimax": 0.15,
            },
            "fal":       {"kling-v2": 0.10, "minimax": 0.08, "luma": 0.06},
            "stability": {"svd": 0.10},
            "huggingface": {"zeroscope": 0.0, "modelscope": 0.0},
        }
        time_map = {
            "replicate": {
                "wan2.1-480p": 45, "wan2.1-i2v-fast": 30,
                "cogvideox": 80, "animatediff": 30,
                "kling-v2.1": 90, "luma-ray": 60,
                "luma-flash": 20, "minimax": 60,
            },
            "fal":       {"kling-v2": 60, "minimax": 45, "luma": 40},
            "stability": {"svd": 90},
            "huggingface": {"zeroscope": 180, "modelscope": 120},
        }

        backend_costs = cost_map.get(backend, {})
        backend_times = time_map.get(backend, {})

        per_clip_cost = max(backend_costs.values()) if not model else backend_costs.get(model, 0.20)
        per_clip_time = max(backend_times.values()) if not model else backend_times.get(model, 60)

        return {
            "num_clips": num_clips,
            "est_cost_usd": round(num_clips * per_clip_cost, 2),
            "est_time_seconds": num_clips * per_clip_time,
            "est_time_minutes": round(num_clips * per_clip_time / 60, 1),
            "backend": backend,
            "model": model,
            "note": "Actual cost/time may vary. Parallel generation reduces wall time.",
        }

    def cleanup(self):
        """Remove all temporary files."""
        try:
            shutil.rmtree(self.output_dir, ignore_errors=True)
        except Exception:
            pass
