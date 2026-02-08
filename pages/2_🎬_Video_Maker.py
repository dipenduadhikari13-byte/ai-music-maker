import streamlit as st
import os
import numpy as np
# import librosa
# from moviepy.editor import VideoClip, AudioFileClip
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
try:
    from PIL import ImageFont
except ImportError:
    ImageFont = None  # Graceful fallback if module unavailable

import requests
from io import BytesIO
import random
import tempfile

# Safe imports for system monitoring
try:
    import psutil
except ImportError:
    psutil = None

try:
    import signal
except ImportError:
    signal = None

# --- CONFIGURATION ---
st.set_page_config(page_title="Music Video Maker", page_icon="🎵", layout="wide")

# Video Settings
WIDTH = 1280
HEIGHT = 720
FPS = 24

# --- MEMORY AND FILE SAFETY ---
def check_system_resources():
    """Monitor system resources to prevent crashes."""
    if not psutil:
        return ["⚠️ Resource monitoring unavailable (install psutil for better safety)"]
    
    try:
        memory_percent = psutil.virtual_memory().percent
        disk_free = psutil.disk_usage('.').free / (1024**3)  # GB
        
        warnings = []
        if memory_percent > 85:
            warnings.append(f"⚠️ High memory usage: {memory_percent:.1f}%")
        if disk_free < 1:
            warnings.append(f"⚠️ Low disk space: {disk_free:.1f}GB free")
        
        return warnings
    except Exception as e:
        return [f"⚠️ Resource check failed: {str(e)}"]

def validate_file_upload(file, max_size_mb=100, allowed_types=None):
    """Validate uploaded files to prevent crashes."""
    if not file:
        return False, "No file uploaded"
    
    # Check file size
    file_size_mb = len(file.getvalue()) / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)"
    
    # Check file type
    if allowed_types and file.type not in allowed_types:
        return False, f"Invalid file type: {file.type}. Allowed: {', '.join(allowed_types)}"
    
    return True, "File valid"

# --- SAFE DEPENDENCY IMPORTS ---
def safe_import_dependencies():
    """Safely import heavy dependencies with graceful fallback."""
    missing_deps = []
    
    try:
        import librosa
        globals()['librosa'] = librosa
    except ImportError as e:
        missing_deps.append(('librosa', 'pip install librosa'))
    
    try:
        from moviepy.editor import VideoClip, AudioFileClip
        globals()['VideoClip'] = VideoClip
        globals()['AudioFileClip'] = AudioFileClip
    except ImportError as e:
        missing_deps.append(('moviepy', 'pip install moviepy'))
    
    if missing_deps:
        st.error("❌ Missing required dependencies:")
        for dep, install_cmd in missing_deps:
            st.code(install_cmd)
        st.info("Please install the missing packages and refresh the page.")
        return False
    
    return True

def resize_image_to_fit(img, target_width, target_height):
    """Resize and crop image to exactly fill the target dimensions"""
    img = img.convert("RGB")
    
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height
    
    if target_ratio > img_ratio:
        # Fit to width
        new_width = target_width
        new_height = int(target_width / img_ratio)
    else:
        # Fit to height
        new_height = target_height
        new_width = int(target_height * img_ratio)
    
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Center crop
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    return img.crop((left, top, right, bottom))

def apply_zoom_effect(img, zoom_factor):
    """Apply zoom effect to image"""
    width, height = img.size
    
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)
    
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    
    cropped = img.crop((left, top, left + new_width, top + new_height))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)

def add_motion_blur(img, intensity=2):
    """Add subtle motion blur"""
    return img.filter(ImageFilter.GaussianBlur(radius=intensity))

def enhance_image(img):
    """Enhance image colors and contrast"""
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    
    # Increase color saturation
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.2)
    
    # Slight brightness boost
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    
    return img

def add_overlay_text(img, text, position, font_size=60, color=(255, 255, 255)):
    """Add text overlay to image"""
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Add shadow for better readability
    shadow_offset = 3
    draw.text((position[0] + shadow_offset, position[1] + shadow_offset), text, font=font, fill=(0, 0, 0, 180))
    draw.text(position, text, font=font, fill=color)
    
    return img

def create_audio_waveform(img, audio_data, sr, current_time, bar_count=50):
    """Create audio-reactive waveform bars at bottom"""
    draw = ImageDraw.Draw(img, 'RGBA')
    width, height = img.size
    
    # Get audio segment for current time
    frame_idx = int(current_time * sr)
    window_size = 2048
    if frame_idx + window_size < len(audio_data):
        segment = audio_data[frame_idx:frame_idx + window_size]
        # Calculate amplitude and ensure it's a scalar
        amplitude = float(np.abs(segment).max())
    else:
        amplitude = 0.0
    
    # Draw bars
    bar_width = width // bar_count
    bar_spacing = 2
    
    for i in range(bar_count):
        # Randomize bar heights based on amplitude
        random_factor = 0.5 + 0.5 * np.random.random()
        bar_height = int((amplitude * 200) * random_factor)
        bar_height = min(bar_height, height // 3)  # Cap at 1/3 screen height
        
        x = i * bar_width
        y = height - bar_height - 20
        
        # Gradient color based on amplitude
        color_intensity = int(255 * amplitude)
        color_intensity = min(255, max(0, color_intensity))  # Clamp to valid range
        bar_color = (color_intensity, 100, 255 - color_intensity, 200)
        
        draw.rectangle(
            [x + bar_spacing, y, x + bar_width - bar_spacing, height - 20],
            fill=bar_color
        )
    
    return img

def add_beat_flash(img, is_beat, intensity=0.3):
    """Add white flash on beats"""
    if is_beat:
        overlay = Image.new('RGBA', img.size, (255, 255, 255, int(255 * intensity)))
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    return img

def add_vignette(img, strength=0.5):
    """Add cinematic vignette effect"""
    width, height = img.size
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Create radial gradient
    center_x, center_y = width // 2, height // 2
    max_radius = int(np.sqrt(center_x**2 + center_y**2))
    
    for radius in range(max_radius, 0, -20):
        alpha = int(255 * strength * (1 - radius / max_radius))
        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            fill=(0, 0, 0, alpha)
        )
    
    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
    return img

def add_particles(img, time, beat_times, particle_count=30):
    """Add floating particles that pulse on beats"""
    draw = ImageDraw.Draw(img, 'RGBA')
    width, height = img.size
    
    # Check if near a beat
    is_near_beat = any(abs(time - float(bt)) < 0.1 for bt in beat_times if abs(time - float(bt)) < 0.5)
    particle_size = 8 if is_near_beat else 4
    
    # Generate particles with pseudo-random but consistent positions
    np.random.seed(int(time * 10))  # Changes every 0.1s
    
    for _ in range(particle_count):
        x = int(np.random.random() * width)
        y = int((np.random.random() * height * 0.8) + (time * 50) % height)  # Drift upward
        alpha = int(np.random.random() * 150) + 50
        
        color = (255, 255, 255, alpha)
        draw.ellipse([x, y, x + particle_size, y + particle_size], fill=color)
    
    return img

def apply_color_grade(img, mood="energetic"):
    """Apply cinematic color grading"""
    if mood == "energetic":
        # Boost saturation, add warmth
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.4)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.1)
    elif mood == "dark":
        # Desaturate, reduce brightness, add contrast
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(0.7)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(0.8)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
    elif mood == "dreamy":
        # Soft, pastel colors
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.2)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.2)
    
    return img

# --- UI LAYOUT ---
st.title("🎵 Music Video Generator")
st.markdown("Upload your song and image to create a professional music video")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Files")
    
    audio_file = st.file_uploader("Upload Audio (MP3/WAV)", type=["mp3", "wav", "m4a"])
    image_file = st.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png"])
    
    st.subheader("🎨 Visual Effects")
    
    effect_type = st.selectbox(
        "Motion Effect",
        ["Slow Zoom In", "Slow Zoom Out", "Ken Burns (Pan & Zoom)", "Pulse Effect", "Static"]
    )
    
    color_mood = st.selectbox(
        "Color Grading",
        ["None", "Energetic (Vibrant)", "Dark (Cinematic)", "Dreamy (Soft)"]
    )
    
    st.subheader("🎵 Audio-Reactive")
    
    audio_reactive = st.checkbox("Audio Waveform Bars", value=True, help="Show animated bars that react to music")
    beat_effects = st.checkbox("Beat Flash Effects", value=True, help="White flashes synced to beats")
    add_particles_fx = st.checkbox("Floating Particles", value=False, help="Aesthetic floating particles")
    add_vignette_fx = st.checkbox("Cinematic Vignette", value=True, help="Dark edges for professional look")
    
    st.subheader("✍️ Text Overlay")
    add_text = st.checkbox("Add Text Overlay")
    
    if add_text:
        overlay_text = st.text_input("Text", placeholder="Song Title or Artist Name")
        text_color = st.color_picker("Text Color", "#FFFFFF")
    
    render_button = st.button("🎬 Generate Viral Video", type="primary", use_container_width=True)

with col2:
    st.subheader("👁️ Preview")
    if image_file:
        preview_img = Image.open(image_file)
        st.image(preview_img, caption="Your Background Image", width=600)
    else:
        st.info("Upload an image to see preview")

# --- 4. PROCESSING LOGIC WITH CRASH PROTECTION ---
if render_button:
    # System resource check
    resource_warnings = check_system_resources()
    for warning in resource_warnings:
        st.warning(warning)
    
    # Input validation
    if not audio_file:
        st.error("❌ Please upload an audio file!")
        st.stop()
    
    if not image_file:
        st.error("❌ Please upload an image file!")
        st.stop()
    
    # File validation
    audio_valid, audio_msg = validate_file_upload(
        audio_file, max_size_mb=50, 
        allowed_types=['audio/mpeg', 'audio/wav', 'audio/mp4']
    )
    if not audio_valid:
        st.error(f"❌ Audio file error: {audio_msg}")
        st.stop()
    
    image_valid, image_msg = validate_file_upload(
        image_file, max_size_mb=20, 
        allowed_types=['image/jpeg', 'image/jpg', 'image/png']
    )
    if not image_valid:
        st.error(f"❌ Image file error: {image_msg}")
        st.stop()
    
    # Dependency check
    if not safe_import_dependencies():
        st.stop()
    
    try:
        with st.spinner("🎧 Initializing Studio Engine..."):
            # Import check with optional timeout (disabled due to platform compatibility)
            try:
                import librosa
                from moviepy.editor import VideoClip, AudioFileClip
                
            except ImportError as e:
                st.error(f"Missing dependency: {e}. Please install required packages.")
                st.code("pip install librosa moviepy pillow")
                st.stop()

        # A. Setup Audio
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tfile.write(audio_file.read())
        audio_path = tfile.name

        try:
            # Load audio with proper error handling
            y, sr = librosa.load(audio_path, sr=None)
            duration = float(librosa.get_duration(y=y, sr=sr))
            
            # Ensure audio data is in the right format
            if y.ndim > 1:
                y = np.mean(y, axis=0)  # Convert stereo to mono if needed
            
            st.success(f"✅ Audio loaded: {duration:.1f}s, {sr}Hz")
            
        except Exception as e:
            st.error(f"Audio File Error: {str(e)}. Please try a different MP3.")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            st.stop()

        # B. Load and prepare image
        base_image = Image.open(image_file)
        base_image = resize_image_to_fit(base_image, WIDTH, HEIGHT)
        
        # Apply color grading
        if color_mood != "None":
            mood_map = {
                "Energetic (Vibrant)": "energetic",
                "Dark (Cinematic)": "dark",
                "Dreamy (Soft)": "dreamy"
            }
            base_image = apply_color_grade(base_image, mood_map[color_mood])
        
        # Add vignette if requested
        if add_vignette_fx:
            base_image = add_vignette(base_image)
        
        # Add text if requested
        if add_text and overlay_text:
            text_position = (50, HEIGHT - 150)
            text_rgb = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            base_image = add_overlay_text(base_image, overlay_text, text_position, color=text_rgb)
        
        st.success("✅ Image processed")
        
        # Detect beats for effects
        st.info("🎵 Analyzing audio for beats...")
        try:
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # Ensure tempo and beat_times are proper numbers
            tempo = float(tempo)
            beat_times = [float(bt) for bt in beat_times]
            
            st.success(f"✅ Detected {len(beat_times)} beats at {tempo:.0f} BPM")
        except Exception as e:
            st.warning(f"Beat detection failed: {str(e)}. Continuing without beat effects.")
            tempo = 120.0
            beat_times = []
        
        # C. Frame generation function
        def make_frame(t):
            try:
                progress = t / duration
                frame = base_image.copy()
                
                # Apply motion effect
                if effect_type == "Slow Zoom In":
                    zoom = 1.0 + (progress * 0.2)
                    frame = apply_zoom_effect(frame, zoom)
                elif effect_type == "Slow Zoom Out":
                    zoom = 1.2 - (progress * 0.2)
                    frame = apply_zoom_effect(frame, zoom)
                elif effect_type == "Ken Burns (Pan & Zoom)":
                    zoom = 1.0 + (0.1 * np.sin(2 * np.pi * progress))
                    frame = apply_zoom_effect(frame, zoom)
                elif effect_type == "Pulse Effect":
                    pulse_freq = 0.5
                    zoom = 1.0 + (0.05 * np.sin(2 * np.pi * pulse_freq * t))
                    frame = apply_zoom_effect(frame, zoom)
                
                # Add particles
                if add_particles_fx:
                    frame = add_particles(frame, t, beat_times)
                
                # Add audio waveform
                if audio_reactive:
                    frame = create_audio_waveform(frame, y, sr, t)
                
                # Add beat flash
                if beat_effects:
                    is_beat = any(abs(t - bt) < 0.05 for bt in beat_times)
                    frame = add_beat_flash(frame, is_beat, intensity=0.2)
                
                # Convert PIL image to numpy array properly
                frame_array = np.array(frame, dtype=np.uint8)
                return frame_array
                
            except Exception as e:
                st.error(f"Frame generation error at time {t}: {str(e)}")
                # Return a blank frame as fallback
                return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        
        # D. Render
        output_filename = "final_render.mp4"
        try:
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
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            if os.path.exists(output_filename):
                os.remove(output_filename)
            
    except ImportError as e:
        st.error(f"❌ Missing required library: {e}")
        st.info("Install requirements: `pip install librosa moviepy pillow streamlit`")
    
    except Exception as e:
        st.error(f"❌ Error generating video: {str(e)}")
        st.exception(e)

# --- FOOTER ---
st.markdown("---")
st.caption("💡 Tip: Use high-resolution images (1920x1080 or larger) for best results")