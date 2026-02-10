"""
Scene Planner — use Gemini LLM to convert lyrics + audio analysis
into a detailed shot-by-shot scene plan for AI video generation.

Each scene gets a rich visual prompt optimised for diffusion video models
(Wan 2.1, CogVideoX, Kling, etc.), including camera motion, style cues,
lighting, and transition type.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class VideoScene:
    """One scene in the music video."""

    scene_id: int
    start_time: float
    end_time: float
    duration: float
    visual_prompt: str          # main prompt for the video model
    negative_prompt: str
    camera_motion: str          # e.g. "slow dolly forward", "orbital pan left"
    style_notes: str            # artistic direction
    transition: str             # crossfade | cut | whip_pan | fade_black
    mood: str
    lyrics_segment: Optional[str] = None
    reference_image_hint: Optional[str] = None


class ScenePlanner:
    """Generate a scene plan using Google Gemini."""

    STYLES = {
        "cinematic":      "Cinematic movie lighting, shallow depth of field, anamorphic lens, film grain, dramatic shadows, 35mm film look",
        "anime":          "High-quality anime style, vibrant colors, detailed cel shading, Studio Ghibli meets Makoto Shinkai, expressive characters",
        "photorealistic": "Photorealistic 8K, natural lighting, shot on RED camera, hyper-detailed textures, photojournalistic",
        "abstract":       "Abstract visual art, flowing colors, geometric patterns, VJ-style visuals, kaleidoscopic, motion graphics",
        "retro":          "Retro 80s VHS aesthetic, neon colors, synthwave, CRT scan lines, analog video effects, Miami Vice vibes",
        "noir":           "Film noir style, high contrast black and white, dramatic shadows, rain-slicked streets, venetian blind lighting",
        "fantasy":        "Epic fantasy world, magical lighting, ethereal atmosphere, Lord-of-the-Rings scale, volumetric fog, enchanted",
        "urban":          "Urban street style, graffiti walls, city lights at night, hypebeast aesthetic, rap music video, tracking shots",
        "nature":         "Breathtaking nature cinematography, Planet Earth style, golden hour, macro details, sweeping aerial shots",
        "surreal":        "Surrealist dreamscape, Salvador Dalí meets digital art, impossible architecture, melting reality, psychedelic",
        "minimal":        "Minimalist aesthetic, clean lines, negative space, monochromatic palette, Scandinavian design, elegant simplicity",
        "bollywood":      "Vibrant Bollywood style, colorful costumes, grand sets, dramatic expressions, rich saturated colors, festive energy",
    }

    CAMERA_MOVES = [
        "slow dolly forward",
        "slow dolly backward",
        "orbital pan left",
        "orbital pan right",
        "crane shot rising",
        "crane shot descending",
        "steady tracking shot",
        "handheld with gentle shake",
        "slow zoom in on subject",
        "dramatic push in",
        "pull back reveal",
        "static wide shot",
        "low angle looking up",
        "high angle looking down",
        "dutch angle tilt",
        "360 rotating shot",
    ]

    TRANSITIONS = ["crossfade", "cut", "whip_pan", "fade_black", "fade_white", "zoom_transition"]

    def __init__(self, gemini_api_key: str, model: str = "gemini-2.0-flash"):
        from google import genai
        self.client = genai.Client(api_key=gemini_api_key)
        self.model = model

    def plan_scenes(
        self,
        audio_analysis,  # AudioAnalysis dataclass
        lyrics: str = "",
        style: str = "cinematic",
        num_scenes: int | None = None,
        custom_instructions: str = "",
        reference_image_descriptions: List[str] | None = None,
    ) -> List[VideoScene]:
        """
        Ask Gemini to create a full scene plan.

        Parameters
        ----------
        audio_analysis : AudioAnalysis
            Output from AudioAnalyzer.analyze().
        lyrics : str
            Song lyrics (can be empty for instrumental).
        style : str
            One of the preset styles or a custom style string.
        num_scenes : int | None
            Override number of scenes. If None, derived from sections.
        custom_instructions : str
            Extra creative direction from the user.
        reference_image_descriptions : list[str] | None
            Descriptions of reference images the user uploaded.

        Returns
        -------
        list[VideoScene]
        """
        style_desc = self.STYLES.get(style, style)
        n_scenes = num_scenes or len(audio_analysis.sections) or max(4, int(audio_analysis.duration / 15))

        # Build section info for the prompt
        sections_text = ""
        for s in audio_analysis.sections:
            sections_text += (
                f"  [{s.start_time:.1f}s – {s.end_time:.1f}s]  "
                f"label={s.label}  energy={s.energy:.2f}  duration={s.duration:.1f}s\n"
            )

        ref_img_text = ""
        if reference_image_descriptions:
            ref_img_text = "Reference images provided by the user:\n"
            for i, desc in enumerate(reference_image_descriptions, 1):
                ref_img_text += f"  Image {i}: {desc}\n"
            ref_img_text += "Incorporate visual elements from these images where appropriate.\n"

        prompt = f"""You are a world-class music video director and cinematographer.
Your job is to plan a shot-by-shot scene breakdown for an AI-generated music video.

=== SONG INFORMATION ===
Duration: {audio_analysis.duration:.1f} seconds
Tempo: {audio_analysis.tempo:.0f} BPM
Key: {audio_analysis.key or 'Unknown'}
Mood: {audio_analysis.mood}
Song sections:
{sections_text}

=== LYRICS ===
{lyrics if lyrics else '(Instrumental — no lyrics)'}

=== VISUAL STYLE ===
{style_desc}

{ref_img_text}

=== CUSTOM DIRECTION ===
{custom_instructions if custom_instructions else 'None — use your best creative judgment.'}

=== TASK ===
Create exactly {n_scenes} scenes that cover the FULL song duration ({audio_analysis.duration:.1f}s).
Each scene's start/end times must be contiguous (scene N ends where scene N+1 starts).
The first scene starts at 0.0s and the last scene ends at {audio_analysis.duration:.1f}s.

For each scene, provide:
1. **visual_prompt**: A detailed, vivid description (2-4 sentences) optimised for AI video
   generation models. Describe the setting, subjects, actions, lighting, colors, textures.
   Be SPECIFIC — avoid vague descriptions. Include the style direction.
   IMPORTANT: Describe MOTION and ACTION, not static images.
2. **negative_prompt**: What to avoid (keep concise).
3. **camera_motion**: One of: {', '.join(self.CAMERA_MOVES[:10])}
4. **style_notes**: Brief artistic direction.
5. **transition**: How this scene transitions to the next. One of: {', '.join(self.TRANSITIONS)}
6. **mood**: Emotional tone of this scene.
7. **lyrics_segment**: The lyrics that play during this scene (if any).

=== OUTPUT FORMAT ===
Return ONLY a valid JSON array. No markdown, no explanation, no code fences.
Each element:
{{
  "scene_id": 1,
  "start_time": 0.0,
  "end_time": 15.5,
  "visual_prompt": "...",
  "negative_prompt": "blurry, low quality, text, watermark, static, still image",
  "camera_motion": "slow dolly forward",
  "style_notes": "...",
  "transition": "crossfade",
  "mood": "...",
  "lyrics_segment": "..."
}}

Make the visual prompts CINEMATIC and DYNAMIC — describe people moving, environments changing,
camera flowing through spaces. This is a REAL video, not a slideshow.
"""

        logger.info("Requesting scene plan from Gemini (%s), %d scenes", self.model, n_scenes)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": 0.8,
                    "max_output_tokens": 8192,
                },
            )
            raw = response.text.strip()
        except Exception as exc:
            logger.error("Gemini scene planning failed: %s", exc)
            return self._fallback_scenes(audio_analysis, style_desc, n_scenes)

        # Parse JSON from response
        scenes = self._parse_response(raw, audio_analysis, style_desc, n_scenes)
        return scenes

    def _parse_response(
        self,
        raw: str,
        audio_analysis,
        style_desc: str,
        n_scenes: int,
    ) -> List[VideoScene]:
        """Parse Gemini JSON response into VideoScene objects."""

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON array in the response
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    logger.error("Cannot parse Gemini response as JSON")
                    return self._fallback_scenes(audio_analysis, style_desc, n_scenes)
            else:
                logger.error("No JSON array found in Gemini response")
                return self._fallback_scenes(audio_analysis, style_desc, n_scenes)

        if not isinstance(data, list):
            data = [data]

        scenes: List[VideoScene] = []
        for i, item in enumerate(data):
            try:
                start = float(item.get("start_time", 0))
                end = float(item.get("end_time", start + 10))
                scenes.append(VideoScene(
                    scene_id=item.get("scene_id", i + 1),
                    start_time=start,
                    end_time=end,
                    duration=round(end - start, 2),
                    visual_prompt=str(item.get("visual_prompt", "")),
                    negative_prompt=str(item.get(
                        "negative_prompt",
                        "blurry, low quality, text, watermark, static, still image",
                    )),
                    camera_motion=str(item.get("camera_motion", "slow dolly forward")),
                    style_notes=str(item.get("style_notes", style_desc)),
                    transition=str(item.get("transition", "crossfade")),
                    mood=str(item.get("mood", audio_analysis.mood)),
                    lyrics_segment=item.get("lyrics_segment"),
                    reference_image_hint=item.get("reference_image_hint"),
                ))
            except (ValueError, TypeError, KeyError) as exc:
                logger.warning("Skipping malformed scene %d: %s", i, exc)

        if not scenes:
            return self._fallback_scenes(audio_analysis, style_desc, n_scenes)

        return scenes

    def _fallback_scenes(
        self,
        audio_analysis,
        style_desc: str,
        n_scenes: int,
    ) -> List[VideoScene]:
        """Generate a reasonable fallback scene plan if Gemini fails."""
        logger.warning("Using fallback scene planner")
        scenes: List[VideoScene] = []
        step = audio_analysis.duration / n_scenes

        moods_prompts = {
            "energetic": [
                "Dynamic aerial shot flying over a glowing neon cityscape at night, lights streaking below, fast-paced energy, camera banking through skyscrapers",
                "Close-up of electric sparks and lightning bolts dancing in slow motion against a dark background, particles swirling with intense color",
                "A crowd of silhouettes dancing under pulsing strobe lights in a massive concert venue, smoke machines creating atmospheric haze",
                "Racing through a tunnel of light with colorful liquid metal walls morphing and flowing, hyper-speed journey through an abstract world",
            ],
            "mellow": [
                "Gentle waves lapping at a golden sand beach during sunset, soft warm light reflecting off the water, peaceful and serene atmosphere",
                "A quiet forest path with sunlight filtering through the canopy, dust motes floating in golden beams, slow camera drift",
                "Soft rain falling on a window pane, blurred city lights visible outside, intimate and contemplative mood, close-up macro shot",
                "A vast field of lavender swaying gently in a warm breeze under a pastel sunset sky, dreamy and ethereal",
            ],
            "dark": [
                "A lone figure walking through a rain-soaked alley with dramatic shadows, neon signs reflecting in puddles, noir atmosphere",
                "Storm clouds rolling over a dark landscape, lightning illuminating the terrain in brief flashes, cinematic and ominous",
                "Abstract dark fluid shapes morphing and swirling in deep blues and blacks, mysterious underwater feeling, bioluminescent accents",
                "An abandoned gothic cathedral interior with dust particles floating through shafts of pale moonlight, hauntingly beautiful",
            ],
            "upbeat": [
                "Colorful confetti and balloons rising in slow motion against a bright blue sky, celebration energy, wide vibrant shot",
                "A group of friends running through a sunlit meadow, camera tracking alongside, golden hour, joyful and carefree",
                "Kaleidoscopic patterns of flowers blooming in time-lapse, rotating and morphing with vivid saturated colors",
                "A surfer riding a crystal-clear turquoise wave, underwater and above-water split shot, tropical paradise feeling",
            ],
            "dramatic": [
                "A massive tidal wave of golden liquid crashing in extreme slow motion, hyper-detailed droplets catching light",
                "An astronaut floating in space with Earth below, visor reflecting starlight, epic scale and solitary beauty",
                "A phoenix rising from swirling embers and flames, wings spreading wide, transforming from fire to brilliant gold light",
                "Mountain landscape at dawn with clouds flowing through valleys like rivers, time-lapse epic scale, majestic",
            ],
        }

        mood = audio_analysis.mood if audio_analysis.mood in moods_prompts else "upbeat"
        prompts = moods_prompts[mood]

        cameras = ["slow dolly forward", "orbital pan left", "crane shot rising",
                    "steady tracking shot", "slow zoom in on subject", "pull back reveal"]
        transitions = ["crossfade", "cut", "crossfade", "fade_black"]

        for i in range(n_scenes):
            start = round(i * step, 2)
            end = round(min((i + 1) * step, audio_analysis.duration), 2)
            prompt_base = prompts[i % len(prompts)]
            full_prompt = f"{prompt_base}. {style_desc}"

            scenes.append(VideoScene(
                scene_id=i + 1,
                start_time=start,
                end_time=end,
                duration=round(end - start, 2),
                visual_prompt=full_prompt,
                negative_prompt="blurry, low quality, text, watermark, static, still image, slideshow",
                camera_motion=cameras[i % len(cameras)],
                style_notes=style_desc,
                transition=transitions[i % len(transitions)],
                mood=mood,
                lyrics_segment=None,
            ))

        return scenes
