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

WIDTH = 1920
HEIGHT = 1080
FPS = 24
BAR_COUNT = 50
BAR_WIDTH = 25
BAR_SPACING = 10
FONT_SIZE = 100

# --- 2. HELPER FUNCTIONS ---

def resize_to_fill(img, target_width, target_height):
    """Resizes and crops an image to fill the screen perfectly (No Black Bars)"""
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height

    if target_ratio > img_ratio:
        # Screen is wider than image: Fit to Width, crop Height
        new_width = target_width
        new_height = int(target_width / img_ratio)
    else:
        # Screen is taller than image: Fit to Height, crop Width
        new_height = target_height
        new_width = int(target_height * img_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Center Crop
    left = (new_width - target_width) / 2
    top = (new_height - target_height) / 2
    right = (new_width + target_width) / 2
    bottom = (new_height + target_height) / 2
    
    return img.crop((left, top, right, bottom))

def get_ai_images(prompt, count=5):
    """Downloads multiple AI images based on the theme"""
    images = []
    progress_bar = st.progress(0, text="🎨 Generating Background Scenes...")
    
    for i in range(count):
        seed = random.randint(1, 999999)
        # Add variation to the prompt to make scenes look different
        variations = ["wide angle", "close up", "cinematic lighting", "atmospheric", "detailed"]
        full_prompt = f"{prompt}, {variations[i % len(variations)]}, 8k wallpaper, highly detailed"
        
        url = f"https://image.pollinations.ai/prompt/{full_prompt}?width={WIDTH}&height={HEIGHT}&seed={seed}&nologo=true"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img = resize_to_fill(img, WIDTH, HEIGHT) # Ensure perfect fit
                images.append(img)
            else:
                st.warning(f"Skipped image {i+1} due to network error.")
        except Exception as e:
            st.error(f"Image Error: {e}")
            
        progress_bar.progress((i + 1) / count)
    
    progress_bar.empty()
    return images

# --- 3. UI LAYOUT ---
st.title("🎬 Professional Music Video Generator")
st.markdown("Turn your Suno MP3s into **YouTube-Ready (1920x1080)** videos with audio visualization.")

col1, col2 = st.columns([1, 2])

with col1:
    st.info("🎵 **Step 1: Upload Song**")
    uploaded_file = st.file_uploader("Choose a Suno MP3", type=["mp3", "wav"])
    
    st.info("🎨 **Step 2: Visual Style**")
    theme = st.text_input("Theme Description", placeholder="e.g., Cyberpunk Samurai in Rain, Neon City")
    if not theme:
        theme = "Abstract cinematic particles, dark background"

    start_btn = st.button("🚀 Render Video", type="primary", use_container_width=True)

with col2:
    preview_area = st.empty()

# --- 4. PROCESSING LOGIC ---
if start_btn and uploaded_file:
    with st.spinner("🎧 Processing Audio & Downloading Visuals..."):
        
        # A. Save Uploaded File Temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tfile.write(uploaded_file.read())
        audio_path = tfile.name

        # B. Analyze Audio
        try:
            y, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            hop = 512
            spectrogram = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
            spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        except Exception as e:
            st.error(f"Audio Error: {e}")
            st.stop()

        # C. Get Images (Cycle through 5)
        bg_images = get_ai_images(theme, count=5)
        if not bg_images:
            st.error("Could not generate images. Please check your internet.")
            st.stop()

        # D. Frame Generator
        def make_frame(t):
            # 1. Determine which image to show based on time
            # Divide song into 5 segments
            segment_duration = duration / len(bg_images)
            img_index = int(t // segment_duration)
            img_index = min(img_index, len(bg_images) - 1) # Safety clamp
            
            frame = bg_images[img_index].copy()
            
            # 2. Beat Pulse Effect
            frame_idx = int(t * sr / hop)
            if frame_idx >= len(onset_env): frame_idx = len(onset_env) - 1
            beat = onset_env[frame_idx]
            
            # Subtle Zoom on Beat
            zoom = 1.0 + (beat * 0.02) 
            w, h = frame.size
            cw, ch = int(w/zoom), int(h/zoom)
            left = (w-cw)//2
            top = (h-ch)//2
            frame = frame.crop((left, top, left+cw, top+ch)).resize((WIDTH, HEIGHT))
            
            draw = ImageDraw.Draw(frame, "RGBA")

            # 3. Audio Bars (Visualizer)
            if frame_idx >= spectrogram_db.shape[1]: frame_idx = spectrogram_db.shape[1] - 1
            db_col = spectrogram_db[:, frame_idx]
            freqs = db_col[:len(db_col)//2] # Use lower half frequencies
            
            chunk = len(freqs) // BAR_COUNT
            total_w = BAR_COUNT * (BAR_WIDTH + BAR_SPACING)
            start_x = (WIDTH - total_w) // 2
            ground_y = HEIGHT - 100
            
            for i in range(BAR_COUNT):
                avg = np.mean(freqs[i*chunk : (i+1)*chunk])
                h_bar = (avg + 80) / 80 * 400 # Scale height
                h_bar = max(5, h_bar * (1 + beat * 0.2)) # React to beat
                
                bx = start_x + i * (BAR_WIDTH + BAR_SPACING)
                by = ground_y - h_bar
                
                # Draw Bar
                draw.rectangle([bx, by, bx+BAR_WIDTH, ground_y], fill=(255, 255, 255, 200))
                # Draw Reflection
                draw.rectangle([bx, ground_y, bx+BAR_WIDTH, ground_y + h_bar*0.2], fill=(255, 255, 255, 50))

            # 4. Progress Bar (Bottom)
            draw.rectangle([0, HEIGHT-10, WIDTH * (t/duration), HEIGHT], fill=(255, 50, 50, 255))
            
            return np.array(frame)

        # E. Render
        output_filename = "final_video.mp4"
        video = VideoClip(make_frame, duration=duration)
        audio_clip = AudioFileClip(audio_path)
        final_video = video.set_audio(audio_clip)

        st.info("⏳ Rendering Video... This may take a minute per minute of song.")
        
        # Write to a temp file for Streamlit
        final_video.write_videofile(
            output_filename, 
            fps=FPS, 
            codec="libx264", 
            audio_codec="aac", 
            threads=4, 
            preset='ultrafast',
            logger=None # Hide console spam
        )

        # F. Display Result
        st.success("✨ Render Complete!")
        st.video(output_filename)
        
        with open(output_filename, "rb") as file:
            st.download_button(
                label="📥 Download Video",
                data=file,
                file_name=f"{theme.replace(' ', '_')}_MusicVideo.mp4",
                mime="video/mp4"
            )
            
        # Cleanup
        os.remove(audio_path)
        os.remove(output_filename)

elif start_btn and not uploaded_file:
    st.warning("⚠️ Please upload a song first!")