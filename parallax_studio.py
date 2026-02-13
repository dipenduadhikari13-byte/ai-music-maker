#!/usr/bin/env python3
"""
Parallax Studio — 2.5-D parallax video from a single still image.

Pipeline
--------
1.  MiDaS (small) estimates a per-pixel depth map on CPU.
2.  OpenCV + NumPy shift every pixel proportionally to its depth
    along a smooth sine-wave trajectory, producing a looping
    2.5-D parallax video.

Dependencies
------------
    pip install torch torchvision timm opencv-python-headless numpy

Usage
-----
    python parallax_studio.py                       # uses input_image.jpg
    python parallax_studio.py -i photo.png -o out.mp4 --duration 6 --fps 30
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


class ParallaxStudio:
    """Generate a depth map with MiDaS and animate a 2.5-D parallax effect."""

    # ------------------------------------------------------------------ #
    #  Construction – load the model once, reuse for many images
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self._device = torch.device("cpu")          # safe for any machine
        print(f"  ⚙  Device : {self._device}")

        print("  📦  Loading MiDaS_small model (first run downloads ~100 MB) …")
        t0 = time.time()
        self._model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", trust_repo=True,
        )
        self._model.to(self._device).eval()

        # The matching transforms shipped with MiDaS
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True,
        )
        self._transform = midas_transforms.small_transform
        print(f"  ✓  Model ready  ({time.time() - t0:.1f}s)\n")

    # ------------------------------------------------------------------ #
    #  Depth estimation
    # ------------------------------------------------------------------ #
    def generate_depth_map(self, image_path: str) -> np.ndarray:
        """Return a float32 depth map normalised to **0 → 1**.

        Higher value  ⇒  **closer** to the camera  ⇒  moves more in
        the parallax animation.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        # --- read & convert ------------------------------------------------
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f"cv2 could not decode: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        print(f"  🖼  Image loaded : {path.name}  ({w}×{h})")

        # --- MiDaS inference ------------------------------------------------
        input_batch = self._transform(rgb).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_batch)

        # Resize prediction to the original image dimensions
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy()             # float32, arbitrary range

        # --- normalise to 0-1 (0 = far, 1 = near) --------------------------
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        print(f"  ✓  Depth map generated  (range 0.00–1.00)")
        return depth.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Parallax animation
    # ------------------------------------------------------------------ #
    def create_animation(
        self,
        image_path: str,
        output_video_path: str = "parallax_output.mp4",
        max_shift_x: int = 30,
        max_shift_y: int = 15,
        duration: float = 5.0,
        fps: int = 30,
    ) -> str:
        """Render a looping 2.5-D parallax video and return its path.

        Parameters
        ----------
        image_path      : source photograph
        output_video_path : destination .mp4 / .avi
        max_shift_x     : maximum horizontal pixel displacement (foreground)
        max_shift_y     : maximum vertical   pixel displacement (foreground)
        duration        : total video length in seconds
        fps             : frames per second
        """
        # --- depth -----------------------------------------------------------
        depth = self.generate_depth_map(image_path)
        bgr = cv2.imread(image_path)
        h, w = bgr.shape[:2]

        total_frames = int(duration * fps)
        print(
            f"\n  🎬  Rendering {total_frames} frames  "
            f"({duration}s @ {fps}fps,  shift ±{max_shift_x}px × ±{max_shift_y}px)"
        )

        # --- build base meshgrid ONCE (vectorised) --------------------------
        #     map_x[y, x] = x,   map_y[y, x] = y   (identity mapping)
        xs = np.arange(w, dtype=np.float32)          # (W,)
        ys = np.arange(h, dtype=np.float32)          # (H,)
        base_x, base_y = np.meshgrid(xs, ys)         # both (H, W)

        # --- crop region (avoid black borders) --------------------------------
        crop_l = max_shift_x
        crop_r = w - max_shift_x
        crop_t = max_shift_y
        crop_b = h - max_shift_y
        out_w = crop_r - crop_l
        out_h = crop_b - crop_t
        if out_w <= 0 or out_h <= 0:
            raise ValueError(
                f"max_shift is too large for this {w}×{h} image.  "
                f"Reduce max_shift_x / max_shift_y."
            )

        # --- video writer -----------------------------------------------------
        ext = Path(output_video_path).suffix.lower()
        if ext in (".mp4", ".m4v"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        elif ext == ".avi":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for: {output_video_path}")

        t0 = time.time()

        for i in range(total_frames):
            # Smooth loop:  0 → 1 → 0 → -1 → 0   (one full sine cycle)
            phase = 2.0 * math.pi * i / total_frames
            offset = math.sin(phase)                 # ∈ [-1, 1]

            # Shift each pixel proportional to its normalised depth.
            # depth=1 (near) gets the full ±max_shift;  depth=0 (far) stays put.
            shift_x = (offset * max_shift_x * depth).astype(np.float32)
            shift_y = (offset * max_shift_y * depth).astype(np.float32)

            map_x = base_x + shift_x
            map_y = base_y + shift_y

            # cv2.remap — vectorised warp, no per-pixel loop
            warped = cv2.remap(
                bgr, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

            # Crop the safe interior region
            frame = warped[crop_t:crop_b, crop_l:crop_r]
            writer.write(frame)

            # progress
            if (i + 1) % fps == 0 or i == total_frames - 1:
                elapsed = time.time() - t0
                pct = (i + 1) / total_frames * 100
                eta = elapsed / (i + 1) * (total_frames - i - 1)
                print(
                    f"\r  ⏳  {i + 1}/{total_frames}  "
                    f"({pct:5.1f}%)   elapsed {elapsed:.1f}s   ETA {eta:.1f}s",
                    end="", flush=True,
                )

        writer.release()
        elapsed = time.time() - t0
        size_mb = os.path.getsize(output_video_path) / 1_048_576
        print(
            f"\n\n  ✅  Video saved: {output_video_path}\n"
            f"      Resolution : {out_w}×{out_h}\n"
            f"      Duration   : {duration}s  ({total_frames} frames @ {fps}fps)\n"
            f"      File size  : {size_mb:.1f} MB\n"
            f"      Render time: {elapsed:.1f}s\n"
        )
        return output_video_path


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="🎞  Parallax Studio — 2.5-D video from a single photo",
    )
    parser.add_argument(
        "-i", "--input", default="input_image.jpg",
        help="Path to source image  (default: input_image.jpg)",
    )
    parser.add_argument(
        "-o", "--output", default="parallax_output.mp4",
        help="Output video path  (default: parallax_output.mp4)",
    )
    parser.add_argument("--max-shift-x", type=int, default=30, help="Max horizontal shift in px")
    parser.add_argument("--max-shift-y", type=int, default=15, help="Max vertical shift in px")
    parser.add_argument("--duration", type=float, default=5.0, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    args = parser.parse_args()

    print(
        "\n╔══════════════════════════════════════════════╗\n"
        "║          🎞  PARALLAX STUDIO                 ║\n"
        "╚══════════════════════════════════════════════╝\n"
    )

    try:
        studio = ParallaxStudio()
        studio.create_animation(
            image_path=args.input,
            output_video_path=args.output,
            max_shift_x=args.max_shift_x,
            max_shift_y=args.max_shift_y,
            duration=args.duration,
            fps=args.fps,
        )
    except FileNotFoundError as exc:
        print(f"\n  ❌  {exc}")
        print("      Please provide a valid image with  -i <path>")
        sys.exit(1)
    except Exception as exc:
        print(f"\n  ❌  Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
