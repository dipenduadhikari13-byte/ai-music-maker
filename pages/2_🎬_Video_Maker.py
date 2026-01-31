import streamlit as st
import os
import numpy as np
import librosa
from moviepy.editor import VideoClip, AudioFileClip
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
        
        # A. Setup Audio
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tfile.write(uploaded_file.read())
        audio_path = tfile.name

        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            hop = 512
            spectrogram = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
            spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
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
            
            # 2. Beat Pulse with Smooth Interpolation
            frame_idx = int(t * sr / hop)
            if frame_idx >= len(onset_env): frame_idx = len(onset_env) - 1
            beat = onset_env[frame_idx]
            beat_smooth = beat * 0.5 + (0.5 if beat > 0.3 else 0)
            
            # Enhanced Zoom Effect
            zoom = 1.0 + (beat_smooth * 0.05) 
            w, h = frame.size
            cw, ch = int(w/zoom), int(h/zoom)
            left = (w-cw)//2
            top = (h-ch)//2
            frame = frame.crop((left, top, left+cw, top+ch)).resize((WIDTH, HEIGHT))
            
            draw = ImageDraw.Draw(frame, "RGBA")

            # 3. ENHANCED Audio Visualizer (Bars with Glow)
            if frame_idx >= spectrogram_db.shape[1]: frame_idx = spectrogram_db.shape[1] - 1
            db_col = spectrogram_db[:, frame_idx]
            freqs = db_col[:len(db_col)//2]
            
            chunk = len(freqs) // BAR_COUNT
            total_w = BAR_COUNT * (BAR_WIDTH + BAR_SPACING)
            start_x = (WIDTH - total_w) // 2
            ground_y = HEIGHT - 100
            
            # CENTER REFLECTION EFFECT
            peak_freq_idx = np.argmax(freqs)
            
            for i in range(BAR_COUNT):
                avg = np.mean(freqs[i*chunk : (i+1)*chunk])
                h_bar = (avg + 80) / 80 * 250
                h_bar = max(5, h_bar * (1 + beat_smooth * 0.4))
                
                bx = start_x + i * (BAR_WIDTH + BAR_SPACING)
                by = ground_y - h_bar
                
                # Frequency-based coloring
                freq_ratio = i / BAR_COUNT
                bar_color = get_frequency_color(freq_ratio)
                
                # Main Bar with gradient
                draw.rectangle([bx, by, bx+BAR_WIDTH, ground_y], fill=(*bar_color, 255))
                # Top Highlight
                draw.rectangle([bx, by, bx+BAR_WIDTH, by+3], fill=(255, 255, 255, 200))
                # Reflection (Lower opacity)
                draw.rectangle([bx, ground_y, bx+BAR_WIDTH, ground_y + h_bar*0.2], fill=(*bar_color, 80))
                
                # Glow effect on peak frequencies
                if abs(i - BAR_COUNT//2) < 5 and beat_smooth > 0.2:
                    apply_glow_effect(draw, bx + BAR_WIDTH//2, by, 8, COLOR_ACCENT, intensity=2)

            # 4. BEAT INDICATOR (Center Circle Pulse)
            center_x, center_y = WIDTH // 2, 60
            pulse_size = int(20 + beat_smooth * 30)
            pulse_color = (255, 200, 0) if beat > 0.3 else (100, 100, 150)
            draw.ellipse(
                [center_x - pulse_size, center_y - pulse_size, center_x + pulse_size, center_y + pulse_size],
                fill=(*pulse_color, 150),
                outline=(255, 255, 255, 100)
            )
            draw.ellipse(
                [center_x - pulse_size//2, center_y - pulse_size//2, center_x + pulse_size//2, center_y + pulse_size//2],
                fill=(255, 255, 255, 80)
            )

            # 5. PROGRESS BAR (Bottom - Studio Style)
            progress_percent = t / duration
            progress_x = WIDTH * progress_percent
            
            # Background track
            draw.rectangle([0, HEIGHT-12, WIDTH, HEIGHT], fill=(40, 40, 60, 200))
            # Progress fill
            draw.rectangle([0, HEIGHT-12, progress_x, HEIGHT], fill=(0, 255, 200, 220))
            # Glowing indicator
            indicator_glow = 8
            for i in range(indicator_glow, 0, -1):
                alpha = int(100 * (1 - i/indicator_glow))
                draw.ellipse(
                    [progress_x-indicator_glow, HEIGHT-12-indicator_glow, progress_x+indicator_glow, HEIGHT+indicator_glow],
                    fill=(0, 255, 200, alpha)
                )

            # 6. TIME DISPLAY
            current_time = int(t)
            total_time = int(duration)
            time_text = f"{current_time//60:02d}:{current_time%60:02d} / {total_time//60:02d}:{total_time%60:02d}"
            
            try:
                # Try to use a bold font if available
                font = ImageFont.truetype("arial.ttf", 18) if os.name == 'nt' else ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
            draw.text((WIDTH-200, 10), time_text, fill=(200, 200, 200, 220), font=font)
            
            # 7. CINEMATIC FILM GRAIN
            frame = add_film_grain(frame, intensity=5)
            
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