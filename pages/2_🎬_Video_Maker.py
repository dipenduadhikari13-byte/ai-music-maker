import streamlit as st
import os
import numpy as np
import librosa
from moviepy.editor import VideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
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
BAR_COUNT = 50
BAR_WIDTH = 20
BAR_SPACING = 8

# --- 2. HELPER FUNCTIONS ---

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
            
            # 2. Beat Pulse
            frame_idx = int(t * sr / hop)
            if frame_idx >= len(onset_env): frame_idx = len(onset_env) - 1
            beat = onset_env[frame_idx]
            
            # Zoom Effect
            zoom = 1.0 + (beat * 0.03) 
            w, h = frame.size
            cw, ch = int(w/zoom), int(h/zoom)
            left = (w-cw)//2
            top = (h-ch)//2
            frame = frame.crop((left, top, left+cw, top+ch)).resize((WIDTH, HEIGHT))
            
            draw = ImageDraw.Draw(frame, "RGBA")

            # 3. Audio Visualizer (Bars)
            if frame_idx >= spectrogram_db.shape[1]: frame_idx = spectrogram_db.shape[1] - 1
            db_col = spectrogram_db[:, frame_idx]
            freqs = db_col[:len(db_col)//2]
            
            chunk = len(freqs) // BAR_COUNT
            total_w = BAR_COUNT * (BAR_WIDTH + BAR_SPACING)
            start_x = (WIDTH - total_w) // 2
            ground_y = HEIGHT - 80
            
            for i in range(BAR_COUNT):
                avg = np.mean(freqs[i*chunk : (i+1)*chunk])
                h_bar = (avg + 80) / 80 * 300 
                h_bar = max(5, h_bar * (1 + beat * 0.3))
                
                bx = start_x + i * (BAR_WIDTH + BAR_SPACING)
                by = ground_y - h_bar
                
                # Glassy White Bars
                draw.rectangle([bx, by, bx+BAR_WIDTH, ground_y], fill=(255, 255, 255, 220))
                draw.rectangle([bx, ground_y, bx+BAR_WIDTH, ground_y + h_bar*0.3], fill=(255, 255, 255, 60))

            # 4. Progress Line
            draw.rectangle([0, HEIGHT-6, WIDTH * (t/duration), HEIGHT], fill=(0, 255, 200, 200))
            
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