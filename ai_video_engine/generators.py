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

    Models
    ------
    wan2.1-720p   high-quality 720p text→video   (~$0.30/clip, ~90 s)
    wan2.1-480p   faster 480p text→video         (~$0.15/clip, ~45 s)
    cogvideox     CogVideoX-5B text→video        (~$0.25/clip, ~80 s)
    animatediff   AnimateDiff (stylised)          (~$0.08/clip, ~30 s)
    """

    T2V_MODELS = {
        "wan2.1-720p":  "wan-ai/wan2.1-t2v-720p",
        "wan2.1-480p":  "wan-ai/wan2.1-t2v-480p",
        "cogvideox":    "tencent/cogvideox-5b",
        "animatediff":  "lucataco/animate-diff",
    }
    I2V_MODELS = {
        "wan2.1-720p":  "wan-ai/wan2.1-i2v-720p-480p",
        "wan2.1-480p":  "wan-ai/wan2.1-i2v-720p-480p",
        "cogvideox":    "tencent/cogvideox-5b",
        "animatediff":  "lucataco/animate-diff",
        "svd":          "stability-ai/stable-video-diffusion-img2vid-xt-1-1",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "wan2.1-720p",
        output_dir: str | None = None,
    ):
        super().__init__(api_key, output_dir)
        self.model_key = model
        self.t2v_id = self.T2V_MODELS.get(model, model)
        self.i2v_id = self.I2V_MODELS.get(model, self.T2V_MODELS.get(model, model))

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

        inp: Dict[str, Any] = {"prompt": prompt}

        if "wan" in self.model_key:
            inp.update(
                negative_prompt=cfg.negative_prompt,
                num_frames=cfg.num_frames,
                guidance_scale=cfg.guidance_scale,
                num_inference_steps=cfg.num_inference_steps,
            )
        elif "cogvideo" in self.model_key:
            inp.update(
                negative_prompt=cfg.negative_prompt,
                num_frames=min(cfg.num_frames, 49),
                guidance_scale=cfg.guidance_scale,
                num_inference_steps=min(cfg.num_inference_steps, 50),
            )
        elif "animatediff" in self.model_key:
            inp.update(
                negative_prompt=cfg.negative_prompt,
                num_frames=min(cfg.num_frames, 32),
                guidance_scale=cfg.guidance_scale,
                num_inference_steps=cfg.num_inference_steps,
            )
        else:
            inp["negative_prompt"] = cfg.negative_prompt

        if cfg.seed >= 0:
            inp["seed"] = cfg.seed
        inp.update(cfg.extra_params)

        if progress_cb:
            progress_cb(f"Submitting to Replicate ({self.model_key})…")

        logger.info("Replicate run  model=%s  prompt=%.100s", self.t2v_id, prompt)

        try:
            output = client.run(self.t2v_id, input=inp)
        except Exception as exc:
            raise RuntimeError(f"Replicate T2V failed: {exc}") from exc

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
        clip_dur = cfg.num_frames / max(cfg.fps, 16)

        return GeneratedClip(
            path=out_path,
            duration=clip_dur,
            width=cfg.width,
            height=cfg.height,
            fps=cfg.fps,
            generation_time=gen_time,
        )

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

        with open(image_path, "rb") as img_fh:
            if "wan" in self.model_key:
                inp: Dict[str, Any] = {
                    "image": img_fh,
                    "prompt": prompt or "Animate this image with natural cinematic motion",
                    "num_frames": cfg.num_frames,
                    "guidance_scale": cfg.guidance_scale,
                    "num_inference_steps": cfg.num_inference_steps,
                }
                if cfg.negative_prompt:
                    inp["negative_prompt"] = cfg.negative_prompt
            else:
                inp = {
                    "input_image": img_fh,
                    "motion_bucket_id": 127,
                    "fps": cfg.fps,
                }

            inp.update(cfg.extra_params)

            if progress_cb:
                progress_cb(f"Submitting I2V to Replicate ({self.i2v_id.split('/')[-1]})…")

            try:
                output = client.run(self.i2v_id, input=inp)
            except Exception as exc:
                raise RuntimeError(f"Replicate I2V failed: {exc}") from exc

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
        clip_dur = cfg.num_frames / max(cfg.fps, 16)

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

    Primary: image→video via SVD.
    Text→video: first generates an image (Stable Image Core), then animates it.
    """

    BASE = "https://api.stability.ai"

    def __init__(self, api_key: str, output_dir: str | None = None):
        super().__init__(api_key, output_dir)

    def is_available(self) -> bool:
        return bool(self.api_key)

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

    Models
    ------
    zeroscope   ZeroScope V2 576×320   (free, lower quality)
    modelscope  ModelScope T2V 1.7B    (free, lower quality)
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
        return bool(self.api_key)

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
            return cls(api_key, model or "wan2.1-720p", output_dir)
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
                "models": ["wan2.1-720p", "wan2.1-480p", "cogvideox", "animatediff"],
                "features": ["text-to-video", "image-to-video"],
                "quality": "★★★★★",
                "speed": "Medium (45–120 s / clip)",
                "cost": "~$0.05–$0.50 per clip",
                "url": "https://replicate.com",
            },
            "fal": {
                "name": "FAL.ai",
                "models": ["kling-v2", "minimax", "luma"],
                "features": ["text-to-video", "image-to-video"],
                "quality": "★★★★★",
                "speed": "Fast (30–90 s / clip)",
                "cost": "~$0.05–$0.15 per clip",
                "url": "https://fal.ai",
            },
            "stability": {
                "name": "Stability AI",
                "models": ["svd"],
                "features": ["image-to-video", "text-to-video (via T2I→I2V)"],
                "quality": "★★★★☆",
                "speed": "Medium (60–120 s / clip)",
                "cost": "~25 credits per clip",
                "url": "https://stability.ai",
            },
            "huggingface": {
                "name": "Hugging Face",
                "models": ["zeroscope", "modelscope"],
                "features": ["text-to-video"],
                "quality": "★★★☆☆",
                "speed": "Slow (60–300 s / clip)",
                "cost": "Free tier available",
                "url": "https://huggingface.co",
            },
        }
