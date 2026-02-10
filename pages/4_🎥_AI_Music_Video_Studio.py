"""
🎥 AI Music Video Studio — Real AI video generation for music videos.

Generate actual motion video using neural network models (Wan 2.1, CogVideoX,
Kling, MiniMax, etc.) — NOT static images, NOT Ken Burns, NOT image slideshows.

Pipeline:
  Song → Audio Analysis → Scene Planning (Gemini AI) → Video Generation → Composition → Music Video
"""

import os
import sys
import time
import json
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional

import streamlit as st

# ── Page Config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Music Video Studio",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("ai_video_studio")

# ── Path setup ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Environment ────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()


def get_key(name: str) -> Optional[str]:
    try:
        k = st.secrets.get(name)
        if k:
            return k
    except Exception:
        pass
    return os.getenv(name)


GEMINI_KEY = get_key("GEMINI_API_KEY") or get_key("GOOGLE_API_KEY")


# ══════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stButton > button { border-radius: 8px; font-weight: 600; }
    .backend-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        color: #e0e0e0;
    }
    .scene-card {
        background: #111827;
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .stat-box {
        background: linear-gradient(135deg, #0f3460, #533483);
        border-radius: 10px;
        padding: 0.7rem 1rem;
        text-align: center;
        color: white;
    }
    .stat-box h3 { margin: 0; font-size: 1.4rem; }
    .stat-box p  { margin: 0; font-size: 0.8rem; opacity: 0.8; }
    div[data-testid="stProgress"] > div > div { height: 6px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════
st.title("🎥 AI Music Video Studio")
st.markdown(
    "Generate **real AI videos** using neural network diffusion models — "
    "Wan 2.1, CogVideoX, Kling, MiniMax, Stable Video Diffusion & more. "
    "Not slideshows, not Ken Burns — **actual motion video** for every scene."
)

# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR — Backend Configuration
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Video Backend")

    # Import engine (lazy — only when needed)
    try:
        from ai_video_engine.generators import VideoGeneratorFactory
        from ai_video_engine.pipeline import MusicVideoPipeline
        from ai_video_engine.scene_planner import ScenePlanner
        ENGINE_OK = True
    except ImportError as exc:
        ENGINE_OK = False
        st.error(f"Engine import error: {exc}")
        st.info("Make sure required packages are installed:\n```\npip install replicate fal-client\n```")

    if ENGINE_OK:
        backends = VideoGeneratorFactory.available_backends()

        backend_choice = st.selectbox(
            "Provider",
            options=list(backends.keys()),
            format_func=lambda k: f"{backends[k]['name']}  {backends[k]['quality']}",
            help="Choose the AI video generation service.",
        )
        bi = backends[backend_choice]

        model_choice = st.selectbox(
            "Model",
            options=bi["models"],
            help=f"Available models on {bi['name']}",
        )

        st.caption(f"**Speed:** {bi['speed']}")
        st.caption(f"**Cost:** {bi['cost']}")
        st.caption(f"[Sign up → {bi['url']}]({bi['url']})")

        st.markdown("---")

        # API Keys — loaded silently from secrets/.env, never displayed
        st.subheader("🔑 API Keys")

        # Load video API key from secrets/env (never pre-fill in UI)
        _env_video_key = get_key(f"{backend_choice.upper()}_API_KEY") or get_key(f"{backend_choice.upper()}_API_TOKEN")

        if _env_video_key:
            st.success(f"✅ {bi['name']} key loaded from environment", icon="🔒")
            video_api_key = _env_video_key
        else:
            st.warning(f"⚠️ No {bi['name']} key found in secrets/.env")
            video_api_key = st.text_input(
                f"{bi['name']} API Key",
                type="password",
                help=f"Get yours at {bi['url']}",
            )

        # Load Gemini key from secrets/env
        if GEMINI_KEY:
            st.success("✅ Gemini key loaded from environment", icon="🔒")
            gemini_key_input = GEMINI_KEY
        else:
            st.info("ℹ️ No Gemini key — will use built-in scene planner")
            gemini_key_input = st.text_input(
                "Gemini API Key (optional)",
                type="password",
                help="Powers the AI scene planner. Get one at aistudio.google.com",
            )

        st.markdown("---")
        st.subheader("🎛️ Advanced")

        max_parallel = st.slider("Parallel generation", 1, 4, 2,
                                  help="Generate multiple clips at once (faster but uses more API quota).")
        clip_fps = st.selectbox("Output FPS", [24, 30, 60], index=0)
        resolution = st.selectbox(
            "Resolution",
            ["1280x720 (720p)", "1920x1080 (1080p)", "854x480 (480p)"],
            index=0,
        )
        res_w, res_h = map(int, resolution.split("(")[0].strip().split("x"))


# ══════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ══════════════════════════════════════════════════════════════════════

if not ENGINE_OK:
    st.stop()

tab_create, tab_about = st.tabs(["🎬 Create Video", "ℹ️ About"])

# ── Tab: Create ───────────────────────────────────────────────────
with tab_create:
    col_input, col_preview = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("📤 Upload & Configure")

        audio_file = st.file_uploader(
            "🎵 Song Audio (required)",
            type=["mp3", "wav", "m4a", "ogg", "flac"],
            help="The song that will become the soundtrack of your music video.",
        )

        lyrics_input = st.text_area(
            "📝 Lyrics (optional but recommended)",
            height=200,
            placeholder="Paste your song lyrics here…\n\nThe AI uses lyrics to create scene descriptions\nthat match the narrative of your song.",
        )

        ref_images = st.file_uploader(
            "🖼️ Reference Images (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            help="Upload images to guide the visual style. The AI will animate these or use them as inspiration.",
        )

        st.markdown("---")
        st.subheader("🎨 Creative Direction")

        style_presets = {
            "cinematic":      "🎬 Cinematic (Film look)",
            "photorealistic": "📷 Photorealistic (8K)",
            "anime":          "🎌 Anime / Animation",
            "abstract":       "🎨 Abstract / VJ Visuals",
            "urban":          "🏙️ Urban / Street",
            "fantasy":        "🧙 Fantasy / Epic",
            "retro":          "📼 Retro / Synthwave",
            "noir":           "🕵️ Film Noir",
            "nature":         "🌿 Nature Documentary",
            "surreal":        "🌀 Surrealist / Dreamscape",
            "minimal":        "◻️ Minimalist",
            "bollywood":      "💃 Bollywood / Vibrant",
        }

        style_choice = st.selectbox(
            "Visual Style",
            options=list(style_presets.keys()),
            format_func=lambda k: style_presets[k],
        )

        custom_direction = st.text_area(
            "✍️ Custom direction (optional)",
            height=80,
            placeholder="e.g. 'Focus on futuristic cityscapes, use lots of neon blue and purple, show a protagonist walking through rain…'",
        )

        col_a, col_b = st.columns(2)
        with col_a:
            num_scenes = st.number_input(
                "Number of scenes",
                min_value=0, max_value=50, value=0,
                help="0 = automatic (based on song structure).",
            )
        with col_b:
            clip_dur_hint = st.number_input(
                "Target clip length (s)",
                min_value=2.0, max_value=15.0, value=5.0, step=0.5,
                help="Approximate length of each AI-generated clip.",
            )

        intro_text = st.text_input("Intro title (optional)", placeholder="Song Title — Artist")
        outro_text = st.text_input("Outro text (optional)", placeholder="Follow @artist")

    # ── Right column: Preview & Output ────────────────────────────
    with col_preview:
        st.subheader("👁️ Preview & Output")

        if audio_file:
            st.audio(audio_file, format=audio_file.type)
        if ref_images:
            cols = st.columns(min(len(ref_images), 4))
            for i, img in enumerate(ref_images):
                with cols[i % len(cols)]:
                    st.image(img, width=200, caption=f"Ref {i + 1}")

        # Cost estimation
        if audio_file:
            st.markdown("---")
            st.subheader("💰 Estimate")
            # Need to know duration — save temp to check
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(audio_file.getvalue())
                tmp_path = tmp.name

            try:
                import librosa
                y, sr = librosa.load(tmp_path, sr=None, mono=True, duration=10)
                duration = librosa.get_duration(path=tmp_path)
                os.remove(tmp_path)
            except Exception:
                duration = 180.0  # estimate 3 min
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            est = MusicVideoPipeline.estimate_cost(
                duration=duration,
                backend=backend_choice,
                model=model_choice,
                clip_duration=clip_dur_hint,
            )

            est_cols = st.columns(4)
            with est_cols[0]:
                st.markdown(f'<div class="stat-box"><h3>{duration:.0f}s</h3><p>Duration</p></div>', unsafe_allow_html=True)
            with est_cols[1]:
                st.markdown(f'<div class="stat-box"><h3>{est["num_clips"]}</h3><p>Clips</p></div>', unsafe_allow_html=True)
            with est_cols[2]:
                cost_str = f"${est['est_cost_usd']:.2f}" if est["est_cost_usd"] > 0 else "Free"
                st.markdown(f'<div class="stat-box"><h3>{cost_str}</h3><p>Est. Cost</p></div>', unsafe_allow_html=True)
            with est_cols[3]:
                st.markdown(f'<div class="stat-box"><h3>{est["est_time_minutes"]:.0f}m</h3><p>Est. Time</p></div>', unsafe_allow_html=True)

        # Video output placeholder
        video_output = st.empty()

    # ── Generate Button ───────────────────────────────────────────
    st.markdown("---")

    gen_col1, gen_col2, gen_col3 = st.columns([1, 2, 1])
    with gen_col2:
        generate_btn = st.button(
            "🎬 Generate AI Music Video",
            type="primary",
            use_container_width=True,
            disabled=not audio_file,
        )

    # ── Generation Logic ──────────────────────────────────────────
    if generate_btn:
        # Validation
        if not audio_file:
            st.error("Upload a song audio file to continue.")
            st.stop()
        if not video_api_key:
            st.error(f"Enter your {bi['name']} API key in the sidebar.")
            st.stop()

        # Prepare files
        work_dir = tempfile.mkdtemp(prefix="mv_studio_")
        audio_path = os.path.join(work_dir, "song" + Path(audio_file.name).suffix)
        with open(audio_path, "wb") as f:
            f.write(audio_file.getvalue())

        ref_img_paths = []
        if ref_images:
            for i, img_file in enumerate(ref_images):
                p = os.path.join(work_dir, f"ref_{i}{Path(img_file.name).suffix}")
                with open(p, "wb") as f:
                    f.write(img_file.getvalue())
                ref_img_paths.append(p)

        # Progress tracking
        progress_bar = st.progress(0.0, text="Initialising…")
        status_text = st.empty()
        scene_expander = st.expander("📋 Scene Plan", expanded=False)
        log_expander = st.expander("📊 Generation Log", expanded=False)
        log_lines = []

        def progress_callback(message: str, pct: float):
            try:
                progress_bar.progress(min(pct, 1.0), text=message)
                status_text.info(f"🎬 {message}")
                log_lines.append(f"[{pct:.0%}] {message}")
                with log_expander:
                    st.code("\n".join(log_lines[-20:]), language="")
            except Exception:
                pass

        try:
            # ── Build Pipeline ────────────────────────────────────
            progress_callback("Initialising AI Video Pipeline…", 0.02)

            pipeline = MusicVideoPipeline(
                video_backend=backend_choice,
                video_api_key=video_api_key,
                video_model=model_choice,
                gemini_api_key=gemini_key_input or GEMINI_KEY,
                output_dir=os.path.join(work_dir, "pipeline"),
                max_parallel=max_parallel,
            )

            # ── Run Generation ────────────────────────────────────
            final_video = pipeline.generate(
                audio_path=audio_path,
                lyrics=lyrics_input,
                style=style_choice,
                resolution=(res_w, res_h),
                fps=clip_fps,
                num_scenes=num_scenes if num_scenes > 0 else None,
                reference_images=ref_img_paths or None,
                custom_instructions=custom_direction,
                clip_duration_hint=clip_dur_hint,
                intro_text=intro_text,
                outro_text=outro_text,
                progress_callback=progress_callback,
            )

            # ── Show Scene Plan ───────────────────────────────────
            if hasattr(pipeline, 'last_scenes') and pipeline.last_scenes:
                try:
                    with scene_expander:
                        for s in pipeline.last_scenes:
                            st.markdown(
                                f'<div class="scene-card">'
                                f'<strong>Scene {s.scene_id}</strong> '
                                f'({s.start_time:.1f}s – {s.end_time:.1f}s) '
                                f'<em>{s.mood}</em><br>'
                                f'📹 {s.camera_motion} &nbsp; 🔄 {s.transition}<br>'
                                f'{s.visual_prompt[:200]}…'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                except Exception:
                    pass  # Scene plan display is optional

            # ── Show Result ───────────────────────────────────────
            progress_bar.progress(1.0, text="✅ Music video complete!")
            st.balloons()

            # Copy to a persistent location
            final_name = f"AI_Music_Video_{int(time.time())}.mp4"
            final_dest = os.path.join(str(ROOT), final_name)
            shutil.copy2(final_video, final_dest)

            with video_output.container():
                st.success(f"✨ Music video generated! ({os.path.getsize(final_dest) / 1_048_576:.1f} MB)")
                st.video(final_dest)
                with open(final_dest, "rb") as vf:
                    st.download_button(
                        "📥 Download Music Video",
                        data=vf,
                        file_name=final_name,
                        mime="video/mp4",
                        use_container_width=True,
                    )

        except Exception as exc:
            progress_bar.progress(0.0, text="❌ Generation failed")
            st.error(f"**Generation Error:** {exc}")
            logger.exception("Pipeline error")
            st.markdown("**Troubleshooting:**")
            st.markdown(f"""
- Verify your **{bi['name']} API key** is correct and has credits
- Check your internet connection
- Try a shorter audio file first
- Try a different model or backend
- Check the error log above for details
""")

        finally:
            # Cleanup temp files (keep the final video)
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass


# ── Tab: About ────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
## How It Works

This studio generates **real AI-generated video** — every scene is created by neural network
video diffusion models that synthesise temporal coherent motion from text or image prompts.

### Pipeline

```
Song Audio ──► Audio Analysis (librosa)
                 │
                 ▼
Lyrics ───────► Scene Planning (Gemini AI)
                 │ Creates N scene descriptions with:
                 │  • Visual prompt (for the video model)
                 │  • Camera motion, mood, transitions
                 ▼
             AI Video Generation
             (Wan 2.1 / CogVideoX / Kling / MiniMax / SVD)
                 │ Generates ~5s of REAL video per scene
                 │ Parallel generation for speed
                 ▼
             Video Composition (ffmpeg)
                 │  • Normalise resolution & FPS
                 │  • Apply transitions (crossfade, cuts, etc.)
                 │  • Mux original song audio
                 ▼
             ★ Final Music Video ★
```

### Supported Backends
""")

    if ENGINE_OK:
        for key, info in VideoGeneratorFactory.available_backends().items():
            with st.expander(f"**{info['name']}** {info['quality']}"):
                st.markdown(f"""
- **Models:** {', '.join(info['models'])}
- **Features:** {', '.join(info['features'])}
- **Speed:** {info['speed']}
- **Cost:** {info['cost']}
- **Sign up:** [{info['url']}]({info['url']})
""")

    st.markdown("""
### What Makes This Different

| Feature | This Studio | Traditional Video Maker |
|---------|------------|------------------------|
| Video type | **AI-generated real motion** | Static images + Ken Burns |
| Per scene | **Neural network renders ~5s video** | Single image with zoom |
| Motion | **Actual object/camera motion** | Fake parallax / pan |
| Variety | **Unique content per scene** | Same image, different crop |
| Quality | **720p-1080p diffusion video** | Image resolution limited |

### Requirements

1. **API Key** for at least one video generation backend
2. **Gemini API Key** for intelligent scene planning (optional but recommended)
3. A song audio file (MP3/WAV)

### Tips

- **Include lyrics** — they dramatically improve scene relevance
- **Upload reference images** — the AI will animate them or match their style
- **Use custom direction** — guide the AI toward your creative vision
- **Start with fewer scenes** for faster iteration
- **Try different backends** — each has a different aesthetic
""")


# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "🎥 AI Music Video Studio • Powered by Wan 2.1, CogVideoX, Kling, Stable Video Diffusion & Gemini AI"
)
