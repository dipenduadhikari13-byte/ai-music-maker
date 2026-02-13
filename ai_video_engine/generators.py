"""
AI Video Generation Backends
=============================
Real AI video generation — NOT static images or Ken Burns effects.

Each backend calls actual neural-network-based video diffusion models
that generate frame-by-frame coherent motion video from text or image prompts.

Supported Backends
------------------
  Replicate     — Wan 2.1 (720p/480p T2V & I2V), CogVideoX-5B, AnimateDiff
  FAL.ai        — Kling Video V2, MiniMax Video, Luma Dream Machine
  Stability AI  — Stable Video Diffusion (I2V, with T2I+I2V pipeline)
  Hugging Face  — ZeroScope V2, ModelScope T2V (free tier)
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data Structures
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class GeneratedClip:
    """Metadata for a single generated video clip."""

    path: str
    duration: float
    width: int = 1280
    height: int = 720
    fps: int = 24
    scene_id: int = -1
    generation_time: float = 0.0


@dataclass
class GenerationConfig:
    """Parameters that control video generation."""

    width: int = 1280
    height: int = 720
    fps: int = 24
    num_frames: int = 81          # ~5 s at 16 fps (Wan 2.1 default)
    guidance_scale: float = 5.0
    num_inference_steps: int = 30
    negative_prompt: str = (
        "blurry, low quality, distorted, ugly, text, watermark, "
        "logo, oversaturated, static, still image, slideshow"
    )
    seed: int = -1
    extra_params: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
#  Abstract Base
# ═══════════════════════════════════════════════════════════════════════


class BaseVideoGenerator(ABC):
    """Interface every video-generation backend must implement."""

    def __init__(self, api_key: str, output_dir: str | None = None):
        self.api_key = api_key
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="aivideo_")
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    # -- public interface ---------------------------------------------------

    @abstractmethod
    def text_to_video(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> GeneratedClip:
        ...

    @abstractmethod
    def image_to_video(
        self,
        image_path: str,
        prompt: str = "",
        config: GenerationConfig | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> GeneratedClip:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def get_name(self) -> str:
        ...

    # -- helpers ------------------------------------------------------------

    def _out(self, prefix: str = "clip") -> str:
        return str(Path(self.output_dir) / f"{prefix}_{uuid.uuid4().hex[:8]}.mp4")

    def _download(self, url: str, dest: str, timeout: int = 600) -> str:
        logger.info("Downloading %s …", url[:120])
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        mb = os.path.getsize(dest) / 1_048_576
        logger.info("Saved %.1f MB → %s", mb, dest)
        return dest

    @staticmethod
    def _extract_url(output) -> str | None:
        """Try hard to get a video URL out of various API response shapes."""
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            for item in output:
                url = BaseVideoGenerator._extract_url(item)
                if url:
                    return url
        if hasattr(output, "url"):
            return output.url
        if isinstance(output, dict):
            for key in ("url", "video", "output", "video_url"):
                if key in output:
                    sub = output[key]
                    if isinstance(sub, str):
                        return sub
                    if isinstance(sub, dict) and "url" in sub:
                        return sub["url"]
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Replicate  (Wan 2.1, CogVideoX, AnimateDiff, SVD)
# ═══════════════════════════════════════════════════════════════════════


class ReplicateGenerator(BaseVideoGenerator):
    """
    Call Replicate-hosted diffusion video models.

    Includes Wan 2.1, CogVideoX, AnimateDiff, plus Kling V2.1, Luma Ray,
    and MiniMax — all available natively on Replicate.

    Models
    ------
    wan2.1-480p     Wan 2.1 480p text→video            (~$0.15/clip, ~45 s)
    wan2.1-i2v-fast Wan 2.2 fast image→video           (~$0.10/clip, ~30 s)
    cogvideox       CogVideoX-5B text→video            (~$0.25/clip, ~80 s)
    animatediff     AnimateDiff (stylised, fast)        (~$0.08/clip, ~30 s)
    kling-v2.1      Kling V2.1 (high quality T2V+I2V)  (~$0.35/clip, ~90 s)
    luma-ray        Luma Ray (cinematic T2V+I2V)        (~$0.30/clip, ~60 s)
    luma-flash      Luma Flash 540p (fast & cheap)      (~$0.06/clip, ~20 s)
    minimax         MiniMax Video-01 (T2V+I2V)          (~$0.15/clip, ~60 s)
    """

    # Model registry: key → {t2v: replicate_model_id, i2v: replicate_model_id | None}
    MODELS = {
        "wan2.1-480p": {
            "t2v": "wan-video/wan-2.1-1.3b",
            "i2v": "wavespeedai/wan-2.1-i2v-480p",
        },
        "wan2.1-i2v-fast": {
            "t2v": "wan-video/wan-2.1-1.3b",
            "i2v": "wan-video/wan-2.2-i2v-fast",
        },
        "cogvideox": {
            "t2v": "cuuupid/cogvideox-5b",
            "i2v": None,
        },
        "animatediff": {
            "t2v": "lucataco/animate-diff",
            "i2v": None,
        },
        "kling-v2.1": {
            "t2v": "kwaivgi/kling-v2.1",
            "i2v": "kwaivgi/kling-v2.1",
        },
        "luma-ray": {
            "t2v": "luma/ray",
            "i2v": "luma/ray",
        },
        "luma-flash": {
            "t2v": "luma/ray-flash-2-540p",
            "i2v": "luma/ray-flash-2-540p",
        },
        "minimax": {
            "t2v": "minimax/video-01",
            "i2v": "minimax/video-01",
        },
    }

    # Default output clip duration per model (seconds)
    CLIP_DURATIONS = {
        "wan2.1-480p": 5.0, "wan2.1-i2v-fast": 5.0,
        "cogvideox": 6.0, "animatediff": 2.0,
        "kling-v2.1": 5.0, "luma-ray": 5.0,
        "luma-flash": 5.0, "minimax": 5.0,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "wan2.1-480p",
        output_dir: str | None = None,
    ):
        super().__init__(api_key, output_dir)
        if model not in self.MODELS:
            logger.warning("Unknown Replicate model '%s', falling back to wan2.1-480p", model)
            model = "wan2.1-480p"
        self.model_key = model
        entry = self.MODELS[model]
        self.t2v_id = entry["t2v"]
        self.i2v_id = entry.get("i2v") or entry["t2v"]

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import replicate  # noqa: F401
            return True
        except ImportError:
            return False

    def get_name(self) -> str:
        return f"Replicate ({self.model_key})"

    # -- input builders (model-specific param mapping) -------------------------

    def _build_t2v_input(self, prompt: str, cfg: GenerationConfig) -> Dict[str, Any]:
        """Build model-specific input parameters for text-to-video."""
        mid = self.t2v_id

        if mid == "wan-video/wan-2.1-1.3b":
            inp = {
                "prompt": prompt,
                "frame_num": cfg.num_frames,
                "sample_steps": min(cfg.num_inference_steps, 30),
                "sample_guide_scale": cfg.guidance_scale,
                "resolution": "480p",
                "aspect_ratio": "16:9",
            }
            if cfg.seed >= 0:
                inp["seed"] = cfg.seed

        elif mid == "cuuupid/cogvideox-5b":
            inp = {
                "prompt": prompt,
                "steps": min(cfg.num_inference_steps, 50),
                "guidance": cfg.guidance_scale,
            }
            if cfg.seed >= 0:
                inp["seed"] = cfg.seed

        elif mid == "lucataco/animate-diff":
            inp = {
                "prompt": prompt,
                "n_prompt": cfg.negative_prompt,
                "steps": cfg.num_inference_steps,
                "guidance_scale": cfg.guidance_scale,
            }
            if cfg.seed >= 0:
                inp["seed"] = cfg.seed

        elif mid == "kwaivgi/kling-v2.1":
            inp = {
                "prompt": prompt,
                "duration": "5",
                "negative_prompt": cfg.negative_prompt,
            }

        elif mid in ("luma/ray", "luma/ray-flash-2-540p"):
            inp = {
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "loop": False,
            }

        elif mid == "minimax/video-01":
            inp = {"prompt": prompt}

        else:
            inp = {"prompt": prompt}

        inp.update(cfg.extra_params)
        return inp

    # -- text → video -------------------------------------------------------

    def text_to_video(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> GeneratedClip:
        import replicate

        cfg = config or GenerationConfig()
        client = replicate.Client(api_token=self.api_key)
        out_path = self._out("rep_t2v")
        t0 = time.time()

        inp = self._build_t2v_input(prompt, cfg)

        if progress_cb:
            progress_cb(f"Submitting to Replicate ({self.model_key})…")

        logger.info("Replicate T2V  model=%s  prompt=%.100s", self.t2v_id, prompt)

        try:
            output = client.run(self.t2v_id, input=inp)
        except Exception as exc:
            raise RuntimeError(f"Replicate T2V failed ({self.t2v_id}): {exc}") from exc

        url = self._extract_url(output)
        if url:
            if progress_cb:
                progress_cb("Downloading generated video…")
            self._download(url, out_path)
        elif hasattr(output, "read"):
            with open(out_path, "wb") as f:
                f.write(output.read())
        else:
            raise RuntimeError(f"Cannot parse Replicate output: {type(output)}")

        gen_time = time.time() - t0
        clip_dur = self.CLIP_DURATIONS.get(self.model_key, 5.0)

        return GeneratedClip(
            path=out_path,
            duration=clip_dur,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            generation_time=gen_time,
        )

    # -- image → video input builder ----------------------------------------

    def _build_i2v_input(self, image_fh, prompt: str, cfg: GenerationConfig) -> Dict[str, Any]:
        """Build model-specific input parameters for image-to-video."""
        mid = self.i2v_id
        default_prompt = prompt or "Animate this image with natural cinematic motion"

        if mid == "wavespeedai/wan-2.1-i2v-480p":
            inp = {
                "image": image_fh,
                "prompt": default_prompt,
                "sample_steps": min(cfg.num_inference_steps, 30),
                "sample_shift": 3.0,
                "aspect_ratio": "16:9",
                "negative_prompt": cfg.negative_prompt,
            }
            if cfg.seed >= 0:
                inp["seed"] = cfg.seed

        elif mid == "wan-video/wan-2.2-i2v-fast":
            inp = {
                "image": image_fh,
                "prompt": default_prompt,
                "num_frames": cfg.num_frames,
                "go_fast": True,
                "resolution": "480p",
            }
            if cfg.seed >= 0:
                inp["seed"] = cfg.seed

        elif mid == "kwaivgi/kling-v2.1":
            inp = {
                "start_image": image_fh,
                "prompt": default_prompt,
                "duration": "5",
                "negative_prompt": cfg.negative_prompt,
            }

        elif mid in ("luma/ray", "luma/ray-flash-2-540p"):
            inp = {
                "start_image": image_fh,
                "prompt": default_prompt,
                "aspect_ratio": "16:9",
                "loop": False,
            }

        elif mid == "minimax/video-01":
            inp = {
                "image": image_fh,
                "prompt": default_prompt,
            }

        else:
            # Fallback — model may not support I2V
            logger.warning("Model %s may not support I2V natively", mid)
            inp = {
                "image": image_fh,
                "prompt": default_prompt,
            }

        inp.update(cfg.extra_params)
        return inp

    # -- image → video ------------------------------------------------------

    def image_to_video(
        self,
        image_path: str,
        prompt: str = "",
        config: GenerationConfig | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> GeneratedClip:
        import replicate

        cfg = config or GenerationConfig()
        client = replicate.Client(api_token=self.api_key)
        out_path = self._out("rep_i2v")
        t0 = time.time()

        # Check if this model supports I2V
        model_entry = self.MODELS.get(self.model_key, {})
        if not model_entry.get("i2v"):
            logger.info("Model %s has no I2V support, using T2V instead", self.model_key)
            return self.text_to_video(
                prompt or "Cinematic animated scene with smooth natural motion",
                cfg, progress_cb,
            )

        with open(image_path, "rb") as img_fh:
            inp = self._build_i2v_input(img_fh, prompt, cfg)

            if progress_cb:
                progress_cb(f"Submitting I2V to Replicate ({self.i2v_id.split('/')[-1]})…")

            logger.info("Replicate I2V  model=%s  prompt=%.100s", self.i2v_id, prompt)

            try:
                output = client.run(self.i2v_id, input=inp)
            except Exception as exc:
                raise RuntimeError(f"Replicate I2V failed ({self.i2v_id}): {exc}") from exc

        url = self._extract_url(output)
        if url:
            if progress_cb:
                progress_cb("Downloading I2V result…")
            self._download(url, out_path)
        elif hasattr(output, "read"):
            with open(out_path, "wb") as f:
                f.write(output.read())
        else:
            raise RuntimeError(f"Cannot parse Replicate I2V output: {type(output)}")

        gen_time = time.time() - t0
        clip_dur = self.CLIP_DURATIONS.get(self.model_key, 5.0)

        return GeneratedClip(
            path=out_path,
            duration=clip_dur,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            generation_time=gen_time,
        )


# ═══════════════════════════════════════════════════════════════════════
#  FAL.ai  (Kling V2, MiniMax, Luma Dream Machine)
# ═══════════════════════════════════════════════════════════════════════


class FalGenerator(BaseVideoGenerator):
    """
    Video generation via FAL.ai hosted models.

    Models
    ------
    kling-v2   Kling Video V2 (5s or 10s, 16:9)   (~$0.10/clip)
    minimax    MiniMax Video 01-Live                (~$0.08/clip)
    luma       Luma Dream Machine                   (~$0.06/clip)
    """

    T2V = {
        "kling-v2": "fal-ai/kling-video/v2/master/text-to-video",
        "minimax":  "fal-ai/minimax-video/video-01-live/text-to-video",
        "luma":     "fal-ai/luma-dream-machine",
    }
    I2V = {
        "kling-v2": "fal-ai/kling-video/v2/master/image-to-video",
        "minimax":  "fal-ai/minimax-video/video-01-live/image-to-video",
        "luma":     "fal-ai/luma-dream-machine/image-to-video",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "kling-v2",
        output_dir: str | None = None,
    ):
        super().__init__(api_key, output_dir)
        self.model_key = model
        self.t2v_ep = self.T2V.get(model, model)
        self.i2v_ep = self.I2V.get(model, self.t2v_ep)

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import fal_client  # noqa: F401
            return True
        except ImportError:
            return False

    def get_name(self) -> str:
        return f"FAL.ai ({self.model_key})"

    def text_to_video(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> GeneratedClip:
        import fal_client

        cfg = config or GenerationConfig()
        out_path = self._out("fal_t2v")
        os.environ["FAL_KEY"] = self.api_key
        t0 = time.time()

        args: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": cfg.negative_prompt,
        }

        if "kling" in self.model_key:
            args["duration"] = "5"
            args["aspect_ratio"] = "16:9"
        elif "minimax" in self.model_key:
            args["prompt_optimizer"] = True
        elif "luma" in self.model_key:
            args["aspect_ratio"] = "16:9"

        args.update(cfg.extra_params)

        if progress_cb:
            progress_cb(f"Generating via FAL.ai ({self.model_key})…")

        def _on_update(update):
            if progress_cb and hasattr(update, "logs") and update.logs:
                for entry in update.logs[-1:]:
                    msg = entry.get("message", "") if isinstance(entry, dict) else str(entry)
                    progress_cb(f"FAL: {msg}")

        try:
            result = fal_client.subscribe(
                self.t2v_ep,
                arguments=args,
                with_logs=True,
                on_queue_update=_on_update,
            )
        except Exception as exc:
            raise RuntimeError(f"FAL T2V failed: {exc}") from exc

        url = self._extract_url(result)
        if not url:
            raise RuntimeError(f"No video URL in FAL response: {result}")

        if progress_cb:
            progress_cb("Downloading FAL video…")
        self._download(url, out_path)

        return GeneratedClip(
            path=out_path,
            duration=5.0,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            generation_time=time.time() - t0,
        )

    def image_to_video(
        self,
        image_path: str,
        prompt: str = "",
        config: GenerationConfig | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> GeneratedClip:
        import fal_client

        cfg = config or GenerationConfig()
        out_path = self._out("fal_i2v")
        os.environ["FAL_KEY"] = self.api_key
        t0 = time.time()

        # Upload image
        if progress_cb:
            progress_cb("Uploading reference image to FAL…")
        try:
            image_url = fal_client.upload_file(image_path)
        except Exception:
            import base64
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            image_url = f"data:image/png;base64,{b64}"

        args: Dict[str, Any] = {
            "prompt": prompt or "Animate this image with natural cinematic motion",
            "image_url": image_url,
        }
        if "kling" in self.model_key:
            args["duration"] = "5"
        args.update(cfg.extra_params)

        if progress_cb:
            progress_cb(f"Generating I2V via FAL ({self.model_key})…")

        try:
            result = fal_client.subscribe(self.i2v_ep, arguments=args, with_logs=True)
        except Exception as exc:
            raise RuntimeError(f"FAL I2V failed: {exc}") from exc

        url = self._extract_url(result)
        if not url:
            raise RuntimeError("No video URL in FAL I2V response")

        self._download(url, out_path)

        return GeneratedClip(
            path=out_path,
            duration=5.0,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            generation_time=time.time() - t0,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Stability AI  (Stable Video Diffusion)
# ═══════════════════════════════════════════════════════════════════════


class StabilityGenerator(BaseVideoGenerator):
    """
    Stability AI REST API.

    ⚠️ DEPRECATED: Stability AI v2beta video endpoints have been removed (404).
    Only SDXL image generation remains available. This backend is kept for
    potential future Stability video API releases.
    """

    BASE = "https://api.stability.ai"

    def __init__(self, api_key: str, output_dir: str | None = None):
        super().__init__(api_key, output_dir)

    def is_available(self) -> bool:
        # v2beta video endpoints return 404 — service discontinued
        logger.warning("Stability AI video endpoints (v2beta) are deprecated and return 404")
        return False

    def get_name(self) -> str:
        return "Stability AI (SVD)"

    # -- text → image (internal) --------------------------------------------

    def _text_to_image(self, prompt: str, cfg: GenerationConfig) -> str:
        r = requests.post(
            f"{self.BASE}/v2beta/stable-image/generate/core",
            headers={"authorization": f"Bearer {self.api_key}", "accept": "image/*"},
            files={"none": ""},
            data={
                "prompt": prompt,
                "negative_prompt": cfg.negative_prompt,
                "aspect_ratio": "16:9",
                "output_format": "png",
            },
            timeout=120,
        )
        if r.status_code != 200:
            raise RuntimeError(f"Stability image gen {r.status_code}: {r.text[:300]}")
        img_path = str(Path(self.output_dir) / f"base_{uuid.uuid4().hex[:8]}.png")
        with open(img_path, "wb") as f:
            f.write(r.content)
        return img_path

    # -- text → video -------------------------------------------------------

    def text_to_video(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> GeneratedClip:
        cfg = config or GenerationConfig()
        if progress_cb:
            progress_cb("Stability: generating base image from prompt…")
        img = self._text_to_image(prompt, cfg)
        return self.image_to_video(img, prompt, cfg, progress_cb)

    # -- image → video ------------------------------------------------------

    def image_to_video(
        self,
        image_path: str,
        prompt: str = "",
        config: GenerationConfig | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> GeneratedClip:
        cfg = config or GenerationConfig()
        out_path = self._out("stab_i2v")
        t0 = time.time()

        if progress_cb:
            progress_cb("Submitting to Stability SVD…")

        with open(image_path, "rb") as fh:
            r = requests.post(
                f"{self.BASE}/v2beta/image-to-video",
                headers={"authorization": f"Bearer {self.api_key}"},
                files={"image": fh},
                data={
                    "seed": cfg.seed if cfg.seed >= 0 else 0,
                    "cfg_scale": cfg.guidance_scale,
                    "motion_bucket_id": 127,
                },
                timeout=120,
            )

        if r.status_code != 200:
            raise RuntimeError(f"Stability SVD submit {r.status_code}: {r.text[:300]}")

        gen_id = r.json().get("id")
        if not gen_id:
            raise RuntimeError("No generation ID from Stability")

        if progress_cb:
            progress_cb("Waiting for Stability AI render…")

        for i in range(180):  # up to 15 min
            res = requests.get(
                f"{self.BASE}/v2beta/image-to-video/result/{gen_id}",
                headers={"authorization": f"Bearer {self.api_key}", "accept": "video/*"},
                timeout=30,
            )
            if res.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(res.content)
                break
            if res.status_code == 202:
                if progress_cb and i % 6 == 0:
                    progress_cb(f"Stability rendering… ({i * 5}s elapsed)")
                time.sleep(5)
                continue
            raise RuntimeError(f"Stability poll error {res.status_code}")
        else:
            raise RuntimeError("Stability AI timed out (15 min)")

        return GeneratedClip(
            path=out_path,
            duration=4.0,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            generation_time=time.time() - t0,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Hugging Face Inference API  (free / pro tier)
# ═══════════════════════════════════════════════════════════════════════


class HuggingFaceGenerator(BaseVideoGenerator):
    """
    Free-tier video generation via HF Inference API.

    ⚠️ DEPRECATED: Both ZeroScope and ModelScope video models have been
    removed from the HuggingFace Inference API (return 410 Gone).
    This backend is kept for potential future HF video model deployments.
    """

    MODELS = {
        "zeroscope":  "cerspense/zeroscope_v2_576w",
        "modelscope": "ali-vilab/text-to-video-ms-1.7b",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "zeroscope",
        output_dir: str | None = None,
    ):
        super().__init__(api_key, output_dir)
        self.model_key = model
        self.model_id = self.MODELS.get(model, model)

    def is_available(self) -> bool:
        # Both video models return 410 Gone — removed from Inference API
        logger.warning("HuggingFace video models (ZeroScope, ModelScope) return 410 Gone")
        return False

    def get_name(self) -> str:
        return f"Hugging Face ({self.model_key})"

    def text_to_video(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> GeneratedClip:
        cfg = config or GenerationConfig()
        out_path = self._out("hf_t2v")
        url = f"https://api-inference.huggingface.co/models/{self.model_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        t0 = time.time()

        if progress_cb:
            progress_cb(f"HF Inference: {self.model_key}…")

        last_err = None
        for attempt in range(4):
            try:
                r = requests.post(url, headers=headers, json={"inputs": prompt}, timeout=300)
            except requests.Timeout:
                raise RuntimeError("HF Inference timed out (5 min)")

            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                break

            if r.status_code == 503:
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                wait = min(body.get("estimated_time", 30), 90)
                if progress_cb:
                    progress_cb(f"HF model loading… retry in {wait:.0f}s (attempt {attempt + 1})")
                time.sleep(wait)
                last_err = f"503 — model loading"
                continue

            last_err = f"{r.status_code}: {r.text[:200]}"
            raise RuntimeError(f"HF API error: {last_err}")
        else:
            raise RuntimeError(f"HF model unavailable after retries: {last_err}")

        return GeneratedClip(
            path=out_path,
            duration=3.0,
            width=576,
            height=320,
            fps=8,
            generation_time=time.time() - t0,
        )

    def image_to_video(
        self,
        image_path: str,
        prompt: str = "",
        config: GenerationConfig | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> GeneratedClip:
        if progress_cb:
            progress_cb("HF free tier: falling back to T2V (I2V not supported)…")
        return self.text_to_video(
            prompt or "Cinematic animated scene with smooth motion",
            config,
            progress_cb,
        )


# ═══════════════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════════════


class VideoGeneratorFactory:
    """Instantiate the right generator from a backend name + API key."""

    _MAP = {
        "replicate":   ReplicateGenerator,
        "fal":         FalGenerator,
        "stability":   StabilityGenerator,
        "huggingface": HuggingFaceGenerator,
    }

    @staticmethod
    def create(
        backend: str,
        api_key: str,
        model: str | None = None,
        output_dir: str | None = None,
    ) -> BaseVideoGenerator:
        backend = backend.lower().strip()
        cls = VideoGeneratorFactory._MAP.get(backend)
        if cls is None:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Choose from: {', '.join(VideoGeneratorFactory._MAP)}"
            )

        if backend == "replicate":
            return cls(api_key, model or "wan2.1-480p", output_dir)
        if backend == "fal":
            return cls(api_key, model or "kling-v2", output_dir)
        if backend == "huggingface":
            return cls(api_key, model or "zeroscope", output_dir)
        # stability — no model param
        return cls(api_key, output_dir)

    @staticmethod
    def available_backends() -> Dict[str, Dict]:
        return {
            "replicate": {
                "name": "Replicate",
                "models": [
                    "wan2.1-480p", "wan2.1-i2v-fast", "cogvideox", "animatediff",
                    "kling-v2.1", "luma-ray", "luma-flash", "minimax",
                ],
                "features": ["text-to-video", "image-to-video"],
                "quality": "★★★★★",
                "speed": "20–120 s / clip (model-dependent)",
                "cost": "~$0.06–$0.35 per clip",
                "url": "https://replicate.com",
            },
            "fal": {
                "name": "FAL.ai ⚠️",
                "models": ["kling-v2", "minimax", "luma"],
                "features": ["text-to-video", "image-to-video"],
                "quality": "★★★★★",
                "speed": "Fast (30–90 s / clip)",
                "cost": "Requires active balance (check fal.ai/dashboard/billing)",
                "url": "https://fal.ai",
            },
            "stability": {
                "name": "Stability AI ❌",
                "models": ["svd"],
                "features": ["DEPRECATED — v2beta video endpoints removed"],
                "quality": "N/A",
                "speed": "N/A",
                "cost": "Video API discontinued",
                "url": "https://stability.ai",
            },
            "huggingface": {
                "name": "Hugging Face ❌",
                "models": ["zeroscope", "modelscope"],
                "features": ["REMOVED — models return 410 Gone"],
                "quality": "N/A",
                "speed": "N/A",
                "cost": "Models removed from Inference API",
                "url": "https://huggingface.co",
            },
        }
