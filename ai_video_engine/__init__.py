"""
AI Video Engine — Real AI-powered video generation for music videos.

Supports multiple generation backends:
  • Replicate  (Wan 2.1, CogVideoX, AnimateDiff)
  • FAL.ai     (Kling V2, MiniMax, Luma Dream Machine)
  • Stability AI (Stable Video Diffusion)
  • Hugging Face Inference API (free tier)

Pipeline:
  Audio Analysis → Scene Planning (Gemini) → AI Video Generation → Composition → Final Music Video
"""

from .generators import (
    VideoGeneratorFactory,
    BaseVideoGenerator,
    ReplicateGenerator,
    FalGenerator,
    StabilityGenerator,
    HuggingFaceGenerator,
    GeneratedClip,
    GenerationConfig,
)
from .scene_planner import ScenePlanner, VideoScene
from .audio_analyzer import AudioAnalyzer, AudioAnalysis, SongSection
from .compositor import VideoCompositor
from .pipeline import MusicVideoPipeline

__all__ = [
    "MusicVideoPipeline",
    "VideoGeneratorFactory",
    "BaseVideoGenerator",
    "ReplicateGenerator",
    "FalGenerator",
    "StabilityGenerator",
    "HuggingFaceGenerator",
    "GeneratedClip",
    "GenerationConfig",
    "ScenePlanner",
    "VideoScene",
    "AudioAnalyzer",
    "AudioAnalysis",
    "SongSection",
    "VideoCompositor",
]
