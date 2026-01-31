import streamlit as st
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import requests
from io import BytesIO
import random
import tempfile

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Pro Video Maker", page_icon="🎬", layout="wide")

# Optimization for Cloud (720p is faster and reliable)
WIDTH = 1280
HEIGHT = 720
FPS = 20
BAR_COUNT = 80  # INCREASED for more detailed visualization
BAR_WIDTH = 12  # ADJUSTED for better density
BAR_SPACING = 3

# Color Palette (CTR-Optimized)
COLOR_BASS = (255, 50, 100)      # Hot Pink/Red (low freq)
COLOR_MID = (100, 200, 255)      # Cyan (mid freq)
COLOR_HIGH = (100, 255, 150)     # Neon Green (high freq)
COLOR_ACCENT = (255, 200, 0)     # Gold (peak moments)

# --- 2. ENHANCED HELPER FUNCTIONS ---

def resize_to_fill(img, target_width, target_height):
    """Resizes and crops an image to fill the screen perfectly"""
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height

    if target_ratio > img_ratio:
        new_width = target_width
        new_height = int(target_width / img_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2
    
    return img.crop((left, top, right, bottom))

def get_ai_images(prompt, count=3):
    """Downloads AI images with a 'Browser Hack' to avoid blocking"""
    images = []
    
    # Fake Browser Headers to bypass simple IP blocks
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    progress_bar = st.progress(0, text="🎨 Fetching AI Backgrounds...")
    
    for i in range(count):
        seed = random.randint(1, 999999)
        # Using 'flux' model often has less strict limits than default
        full_prompt = f"{prompt}, cinematic lighting, 4k wallpaper, detailed --ar 16:9"
        url = f"https://image.pollinations.ai/prompt/{full_prompt}?width={WIDTH}&height={HEIGHT}&seed={seed}&nologo=true&model=flux"
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = resize_to_fill(img, WIDTH, HEIGHT)
                images.append(img)
            else:
                st.warning(f"⚠️ Image {i+1} failed to load. Using placeholder.")
                # Fallback to a solid color if AI fails
                images.append(Image.new('RGB', (WIDTH, HEIGHT), color = (20, 20, 40)))
        except Exception as e:
            st.error(f"Connection Error: {e}")
            images.append(Image.new('RGB', (WIDTH, HEIGHT), color = (20, 20, 40)))
            
        progress_bar.progress((i + 1) / count)
    
    progress_bar.empty()
    return images

# --- 2.5. STUDIO ENHANCEMENT FUNCTIONS ---

def enhance_contrast(img, factor=1.2):
    """Enhance image contrast for more vibrant visuals"""
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)

def enhance_saturation(img, factor=1.3):
    """Boost color saturation for cinematic look"""
    enhancer = ImageEnhance.Color(img)
    return enhancer.enhance(factor)

def add_vignette(img, intensity=0.3):
    """Add dark vignette effect around edges"""
    width, height = img.size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Create radial gradient
    for i in range(min(width, height) // 2):
        alpha = int(255 * (1 - (i / (min(width, height) / 2)) ** 2) * intensity)
        draw.ellipse(
            [width//2 - i, height//2 - i, width//2 + i, height//2 + i],
            fill=alpha
        )
    
    # Apply mask
    dark_overlay = Image.new('RGB', (width, height), (0, 0, 0))
    return Image.composite(img, dark_overlay, mask)

def get_frequency_color(freq_ratio):
    """Return color based on frequency range (bass=red, mid=cyan, high=green)"""
    if freq_ratio < 0.33:  # Bass
        return COLOR_BASS
    elif freq_ratio < 0.66:  # Mids
        return COLOR_MID
    else:  # Highs
        return COLOR_HIGH

def apply_glow_effect(draw, x, y, radius, color, intensity=1):
    """Draw a glowing effect at specified position"""
    for i in range(radius, 0, -1):
        alpha = int(80 * (1 - i/radius) * intensity)
        draw.ellipse(
            [x - i, y - i, x + i, y + i],
            fill=(*color, alpha)
        )

def add_film_grain(img, intensity=5):
    """Add subtle film grain for cinematic effect"""
    width, height = img.size
    pixels = np.array(img)
    
    # Generate random noise
    noise = np.random.randint(-intensity, intensity, (height, width, 3), dtype=np.int16)
    
    # Add noise to image
    noisy = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(noisy)

# --- 3. UI LAYOUT ---
st.title("🎬 Pro Music Video Studio")
st.markdown("Create professional visualizations for your Suno tracks.")

col1, col2 = st.columns([1, 2])

with col1:
    st.success("📂 **Step 1: Assets**")
    uploaded_file = st.file_uploader("1. Upload Song (MP3/WAV)", type=["mp3", "wav"])
    
    # NEW: Custom Background Uploader (The Fix!)
    bg_upload = st.file_uploader("2. Upload Background (Optional - Fixes AI Error)", type=["jpg", "png", "jpeg"])
    
    st.info("🎨 **Step 2: Style (If no background uploaded)**")
    theme = st.text_input("AI Prompt", placeholder="e.g. Cyberpunk samurai in rain")
    if not theme: theme = "Abstract nebula, dark space, cinematic"

    start_btn = st.button("🚀 Render Video", type="primary", use_container_width=True)

with col2:
    st.caption("Preview Area")
    if bg_upload:
        st.image(bg_upload, caption="Using Custom Background", use_column_width=True)

# --- 4. PROCESSING LOGIC ---
if start_btn and uploaded_file:
    with st.spinner("🎧 Initializing Studio Engine..."):
        try:
            import librosa
            from moviepy.editor import VideoClip, AudioFileClip
        except Exception as e:
            st.error(f"Missing dependency: {e}. Please install required packages.")
            st.stop()

        # A. Setup Audio
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tfile.write(uploaded_file.read())
        audio_path = tfile.name

        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
        except Exception as e:
            st.error("Audio File Error. Please try a different MP3.")
            st.stop()

        # B. Get Visuals (The Smart Selection)
        bg_images = []
        
        if bg_upload:
            # OPTION 1: USER UPLOAD (100% Reliability)
            img = Image.open(bg_upload).convert("RGB")
            img = resize_to_fill(img, WIDTH, HEIGHT)
            bg_images = [img] * 3 # Use same image or dupes
            st.toast("✅ Using Custom Background")
        else:
            # OPTION 2: AI GENERATION (With Header Hack)
            st.toast("🤖 Generating AI Scenes...")
            bg_images = get_ai_images(theme, count=3)

        if not bg_images:
            st.error("Failed to load any images.")
            st.stop()

        # C. Define Frame Generator
        def make_frame(t):
            # 1. Switch Image
            segment_duration = duration / len(bg_images)
            img_index = int(t // segment_duration)
            img_index = min(img_index, len(bg_images) - 1)
            
            frame = bg_images[img_index].copy()
            
            # STUDIO QUALITY: Enhance the background
            frame = enhance_contrast(frame, factor=1.2)
            frame = enhance_saturation(frame, factor=1.3)
            frame = add_vignette(frame, intensity=0.3)
            
            # Simple cinematic slow zoom (no beat, no bars)
            zoom = 1.0 + (0.01 * np.sin(2 * np.pi * t / 8.0))
            w, h = frame.size
            cw, ch = int(w / zoom), int(h / zoom)
            left = (w - cw) // 2
            top = (h - ch) // 2
            frame = frame.crop((left, top, left + cw, top + ch)).resize((WIDTH, HEIGHT))
            
            # Cinematic film grain
            frame = add_film_grain(frame, intensity=4)
            
            return np.array(frame)

        # D. Render
        output_filename = "final_render.mp4"
        video = VideoClip(make_frame, duration=duration)
        audio_clip = AudioFileClip(audio_path)
        final_video = video.set_audio(audio_clip)

        st.info(f"⏳ Rendering Video ({int(duration)}s)... Please wait.")
        
        final_video.write_videofile(
            output_filename, 
            fps=FPS, 
            codec="libx264", 
            audio_codec="aac", 
            threads=4, 
            preset='ultrafast',
            logger=None
        )

        st.balloons()
        st.success("✨ Video Ready!")
        st.video(output_filename)
        
        with open(output_filename, "rb") as file:
            st.download_button("📥 Download MP4", data=file, file_name="Suno_Visualizer.mp4", mime="video/mp4")
            
        os.remove(audio_path)
        os.remove(output_filename)

elif start_btn:
    st.warning("Please upload a song first.")