import streamlit as st
import os
import gc
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import tempfile
import zipfile
from io import BytesIO

# Safe imports
try:
    import librosa
except ImportError:
    librosa = None

try:
    from moviepy.editor import VideoClip, AudioFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip
except ImportError:
    VideoClip = None

# --- CONFIGURATION ---
st.set_page_config(page_title="Multi-Image Music Video", page_icon="🎥", layout="wide")

WIDTH = 1920
HEIGHT = 1080
FPS = 30

# --- HELPER FUNCTIONS ---

def validate_dependencies():
    """Check if required libraries are installed"""
    missing = []
    if not librosa:
        missing.append("librosa")
    if not VideoClip:
        missing.append("moviepy")
    
    if missing:
        st.error(f"❌ Missing dependencies: {', '.join(missing)}")
        st.code(f"pip install {' '.join(missing)}")
        return False
    return True

def resize_image_to_fit(img, target_width, target_height):
    """Resize and crop image to exactly fill the target dimensions"""
    img = img.convert("RGB")
    
    img_ratio = img.width / img.height
    target_ratio = target_width / target_height
    
    if target_ratio > img_ratio:
        new_width = target_width
        new_height = int(target_width / img_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * img_ratio)
    
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    
    return img.crop((left, top, right, bottom))

def apply_ken_burns(img, progress, effect_type="zoom_in"):
    """Apply Ken Burns effect (pan and zoom)"""
    width, height = img.size
    
    if effect_type == "zoom_in":
        zoom = 1.0 + (progress * 0.15)
    elif effect_type == "zoom_out":
        zoom = 1.15 - (progress * 0.15)
    elif effect_type == "pan_right":
        zoom = 1.1
        # Shift the crop window
        shift = int(width * 0.1 * progress)
        img = img.crop((shift, 0, width, height))
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return img
    else:
        zoom = 1.0
    
    new_width = int(width / zoom)
    new_height = int(height / zoom)
    
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    
    cropped = img.crop((left, top, left + new_width, top + new_height))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)

def create_beat_visualizer(width, height, audio_data, sr, current_time, beat_times):
    """Create circular beat visualizer overlay"""
    visualizer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(visualizer)
    
    # Get audio amplitude at current time
    frame_idx = int(current_time * sr)
    window_size = 2048
    
    if frame_idx + window_size < len(audio_data):
        segment = audio_data[frame_idx:frame_idx + window_size]
        amplitude = float(np.abs(segment).max())
    else:
        amplitude = 0.0
    
    # Check if near a beat
    is_beat = any(abs(current_time - float(bt)) < 0.1 for bt in beat_times)
    
    # Circular visualizer in center
    center_x, center_y = width // 2, height // 2
    base_radius = 150
    
    # Draw circular bars
    num_bars = 60
    for i in range(num_bars):
        angle = (2 * np.pi * i / num_bars) - np.pi/2
        
        # Random variation per bar
        np.random.seed(int(current_time * 100) + i)
        bar_height = int(base_radius * amplitude * 2 * (0.5 + 0.5 * np.random.random()))
        bar_height = min(bar_height, 200)
        
        # Pulse on beats
        if is_beat:
            bar_height = int(bar_height * 1.5)
        
        # Calculate positions
        inner_radius = base_radius
        outer_radius = base_radius + bar_height
        
        x1 = int(center_x + inner_radius * np.cos(angle))
        y1 = int(center_y + inner_radius * np.sin(angle))
        x2 = int(center_x + outer_radius * np.cos(angle))
        y2 = int(center_y + outer_radius * np.sin(angle))
        
        # Color gradient
        color_intensity = int(255 * amplitude)
        color_intensity = min(255, max(0, color_intensity))
        bar_color = (color_intensity, 150, 255 - color_intensity, 200)
        
        draw.line([x1, y1, x2, y2], fill=bar_color, width=4)
    
    # Center circle glow
    if is_beat:
        glow_radius = int(base_radius * 0.7)
        draw.ellipse(
            [center_x - glow_radius, center_y - glow_radius,
             center_x + glow_radius, center_y + glow_radius],
            fill=(255, 255, 255, 150)
        )
    
    return visualizer

def create_waveform_bars(width, height, audio_data, sr, current_time, bar_count=80):
    """Create bottom waveform bars"""
    waveform = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(waveform)
    
    frame_idx = int(current_time * sr)
    window_size = 2048
    
    if frame_idx + window_size < len(audio_data):
        segment = audio_data[frame_idx:frame_idx + window_size]
        amplitude = float(np.abs(segment).max())
    else:
        amplitude = 0.0
    
    bar_width = width // bar_count
    bar_spacing = 2
    
    for i in range(bar_count):
        np.random.seed(int(current_time * 50) + i)
        random_factor = 0.3 + 0.7 * np.random.random()
        bar_height = int((amplitude * 300) * random_factor)
        bar_height = min(bar_height, height // 4)
        
        x = i * bar_width
        y = height - bar_height - 40
        
        # Gradient color
        color_intensity = int(255 * amplitude)
        color_intensity = min(255, max(0, color_intensity))
        bar_color = (255, color_intensity, 100, 220)
        
        draw.rectangle(
            [x + bar_spacing, y, x + bar_width - bar_spacing, height - 40],
            fill=bar_color
        )
    
    return waveform

def apply_transition_effect(img1, img2, progress, effect="fade"):
    """Apply transition between two images"""
    if effect == "fade":
        # Simple crossfade
        img1_array = np.array(img1, dtype=np.float32)
        img2_array = np.array(img2, dtype=np.float32)
        blended = img1_array * (1 - progress) + img2_array * progress
        return Image.fromarray(blended.astype(np.uint8))
    
    elif effect == "slide_left":
        width, height = img1.size
        offset = int(width * progress)
        result = Image.new('RGB', (width, height))
        result.paste(img1, (-offset, 0))
        result.paste(img2, (width - offset, 0))
        return result
    
    elif effect == "zoom_blur":
        if progress < 0.5:
            return img1.filter(ImageFilter.GaussianBlur(radius=progress * 20))
        else:
            return img2.filter(ImageFilter.GaussianBlur(radius=(1 - progress) * 20))
    
    return img1

def add_cinematic_bars(img, bar_height=80):
    """Add black cinematic bars top and bottom"""
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Top bar
    draw.rectangle([0, 0, width, bar_height], fill=(0, 0, 0, 255))
    # Bottom bar
    draw.rectangle([0, height - bar_height, width, height], fill=(0, 0, 0, 255))
    
    return img

def enhance_image_quality(img):
    """Apply professional color grading"""
    # Increase contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # Increase sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.3)
    
    # Slight saturation boost
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(1.15)
    
    return img

# --- UI LAYOUT ---
st.title("🎥 Studio-Grade Multi-Image Music Video")
st.markdown("Upload one audio file and 6-12 images to create a professional music video with beat visualizer")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Files")
    
    audio_file = st.file_uploader("Upload Audio (MP3/WAV)", type=["mp3", "wav", "m4a"])
    
    image_files = st.file_uploader(
        "Upload 6-12 High Quality Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload images in the order you want them to appear"
    )
    
    if image_files:
        st.success(f"✅ {len(image_files)} images uploaded")
        if len(image_files) < 6:
            st.warning("⚠️ Upload at least 6 images for best results")
        elif len(image_files) > 12:
            st.warning("⚠️ Using first 12 images only (recommended limit)")
            image_files = image_files[:12]
    
    st.subheader("🎨 Video Settings")
    
    transition_style = st.selectbox(
        "Transition Effect",
        ["Fade", "Slide Left", "Zoom Blur"]
    )
    
    motion_effect = st.selectbox(
        "Image Motion",
        ["Ken Burns Zoom In", "Ken Burns Zoom Out", "Pan Right", "Static"]
    )
    
    visualizer_style = st.selectbox(
        "Beat Visualizer",
        ["Circular (Center)", "Waveform Bars (Bottom)", "Both", "None"]
    )
    
    add_cinematic = st.checkbox("Cinematic Black Bars", value=True, help="Add professional letterbox bars")
    
    auto_timing = st.checkbox("Auto-Sync to Beats", value=True, help="Change images on detected beats")
    
    if not auto_timing:
        image_duration = st.slider("Seconds per Image", 2.0, 8.0, 4.0, 0.5)
    
    render_quality = st.selectbox(
        "Render Quality",
        ["High (1080p)", "Ultra (1080p 60fps)", "Medium (720p)"]
    )
    
    render_button = st.button("🎬 Generate Studio Video", type="primary", use_container_width=True)

with col2:
    st.subheader("📸 Image Preview")
    if image_files and len(image_files) > 0:
        # Show first 3 images as preview
        preview_cols = st.columns(min(3, len(image_files)))
        for idx, img_file in enumerate(image_files[:3]):
            with preview_cols[idx]:
                preview_img = Image.open(img_file)
                st.image(preview_img, caption=f"Image {idx+1}", use_container_width=True)
        
        if len(image_files) > 3:
            st.caption(f"+ {len(image_files) - 3} more images...")
    else:
        st.info("Upload images to see preview")

# --- PROCESSING ---
if render_button:
    if not audio_file:
        st.error("❌ Please upload an audio file!")
        st.stop()
    
    if not image_files or len(image_files) < 6:
        st.error("❌ Please upload at least 6 images!")
        st.stop()
    
    if not validate_dependencies():
        st.stop()
    
    try:
        import librosa
        from moviepy.editor import VideoClip, AudioFileClip, ImageClip, concatenate_videoclips, CompositeVideoClip
        
        # Set render settings
        if render_quality == "Ultra (1080p 60fps)":
            fps = 60
            width, height = 1920, 1080
        elif render_quality == "High (1080p)":
            fps = 30
            width, height = 1920, 1080
        else:
            fps = 24
            width, height = 1280, 720
        
        with st.spinner("🎧 Analyzing audio..."):
            # Save audio to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tfile.write(audio_file.read())
            audio_path = tfile.name
            tfile.close()
            
            # Load audio at low sample rate for beat detection (saves memory)
            y_beats, sr_beats = librosa.load(audio_path, sr=16000, mono=True, dtype=np.float32)
            duration = float(librosa.get_duration(y=y_beats, sr=sr_beats))
            
            st.success(f"✅ Audio: {duration:.1f}s")
        
        with st.spinner("🎵 Detecting beats..."):
            # Use smaller hop_length for memory efficiency
            tempo, beat_frames = librosa.beat.beat_track(y=y_beats, sr=sr_beats, hop_length=512)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr_beats, hop_length=512)
            
            # Free memory immediately
            del y_beats
            gc.collect()
            
            # Load audio at LOW sample rate for visualizer (16kHz is enough for amplitude)
            y, sr = librosa.load(audio_path, sr=16000, mono=True, dtype=np.float32)
            
            # Convert to proper Python types
            tempo = float(tempo) if hasattr(tempo, '__float__') else float(tempo[0]) if hasattr(tempo, '__iter__') else 120.0
            beat_times = [float(bt) for bt in beat_times]
            
            st.success(f"✅ Detected {len(beat_times)} beats at {tempo:.0f} BPM")
        
        with st.spinner("🖼️ Processing images..."):
            # Save processed images as compressed JPEG bytes (much smaller than numpy arrays)
            processed_image_bytes = []
            for img_file in image_files:
                img = Image.open(img_file)
                img = resize_image_to_fit(img, width, height)
                img = enhance_image_quality(img)
                if add_cinematic:
                    img = add_cinematic_bars(img)
                # Store as compressed JPEG bytes
                buf = BytesIO()
                img.convert('RGB').save(buf, format='JPEG', quality=90)
                processed_image_bytes.append(buf.getvalue())
                del img
            gc.collect()
            
            num_images = len(processed_image_bytes)
            st.success(f"✅ Processed {num_images} images")
        
        # Calculate image timing
        if auto_timing and len(beat_times) > num_images:
            # Distribute images across beats
            beats_per_image = len(beat_times) // num_images
            image_change_times = [beat_times[i * beats_per_image] for i in range(num_images)]
            image_change_times.append(duration)
        else:
            # Equal duration for each image
            img_duration = duration / num_images
            image_change_times = [i * img_duration for i in range(num_images + 1)]
        
        st.info(f"📊 Timeline: {num_images} images over {duration:.1f}s")
        
        # Determine transition effect
        transition_map = {
            "Fade": "fade",
            "Slide Left": "slide_left",
            "Zoom Blur": "zoom_blur"
        }
        transition_type = transition_map[transition_style]
        
        motion_map = {
            "Ken Burns Zoom In": "zoom_in",
            "Ken Burns Zoom Out": "zoom_out",
            "Pan Right": "pan_right",
            "Static": "static"
        }
        motion_type = motion_map[motion_effect]
        
        # Keep audio as numpy array (already float32, memory efficient)
        audio_data = y  # Don't convert to list - keep as numpy
        audio_sr = int(sr)
        beat_times_list = list(beat_times)
        change_times_list = list(image_change_times)
        total_duration = float(duration)
        
        # Free processed_images list if it exists
        gc.collect()
        
        st.info(f"🎬 Starting render: {total_duration:.1f}s video...")
        
        # Create frame generator with compressed images
        class FrameGenerator:
            def __init__(self, image_bytes, audio, sample_rate, beats, changes, dur, motion, transition, viz_style, w, h):
                self.image_bytes = image_bytes  # Compressed JPEG bytes
                self.image_cache = {}  # Cache decoded images
                self.audio = audio  # Keep as numpy array
                self.sr = sample_rate
                self.beats = beats
                self.changes = changes
                self.duration = dur
                self.motion = motion
                self.transition = transition
                self.viz_style = viz_style
                self.width = w
                self.height = h
                self.last_progress = 0
            
            def _get_image(self, idx):
                """Load image from compressed bytes, with simple caching"""
                if idx not in self.image_cache:
                    # Keep cache small - only current and next image
                    if len(self.image_cache) >= 2:
                        self.image_cache.clear()
                    img = Image.open(BytesIO(self.image_bytes[idx]))
                    self.image_cache[idx] = img
                return self.image_cache[idx].copy()
            
            def get_frame(self, t):
                try:
                    # Find current image index
                    current_idx = 0
                    for idx in range(len(self.changes) - 1):
                        if t >= self.changes[idx]:
                            current_idx = idx
                    
                    # Ensure valid index
                    current_idx = min(current_idx, len(self.image_bytes) - 1)
                    
                    # Get base image from compressed bytes
                    frame = self._get_image(current_idx)
                    
                    # Calculate segment progress
                    seg_start = self.changes[current_idx]
                    seg_end = self.changes[min(current_idx + 1, len(self.changes) - 1)]
                    seg_dur = max(seg_end - seg_start, 0.1)
                    progress = min(max((t - seg_start) / seg_dur, 0), 1)
                    
                    # Apply motion effect
                    if self.motion != "static":
                        frame = apply_ken_burns(frame, progress, self.motion)
                    
                    # Apply transition near segment end
                    trans_dur = 0.5
                    time_to_end = seg_end - t
                    if time_to_end < trans_dur and current_idx < len(self.image_bytes) - 1:
                        next_frame = self._get_image(current_idx + 1)
                        if self.motion != "static":
                            next_frame = apply_ken_burns(next_frame, 0, self.motion)
                        trans_progress = min(max((trans_dur - time_to_end) / trans_dur, 0), 1)
                        frame = apply_transition_effect(frame, next_frame, trans_progress, self.transition)
                    
                    # Convert to RGBA for overlays
                    frame = frame.convert('RGBA')
                    
                    # Add visualizers
                    if self.viz_style in ["Circular (Center)", "Both"]:
                        viz = self._create_circular_viz(t)
                        frame = Image.alpha_composite(frame, viz)
                    
                    if self.viz_style in ["Waveform Bars (Bottom)", "Both"]:
                        wave = self._create_waveform(t)
                        frame = Image.alpha_composite(frame, wave)
                    
                    # Convert to RGB numpy array
                    frame = frame.convert('RGB')
                    return np.array(frame, dtype=np.uint8)
                    
                except Exception as e:
                    # Return blank frame on error
                    return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            def _get_amplitude(self, t):
                frame_idx = int(t * self.sr)
                window = 2048
                if frame_idx + window < len(self.audio):
                    segment = self.audio[frame_idx:frame_idx + window]
                    return float(np.abs(segment).max())
                return 0.0
            
            def _create_circular_viz(self, t):
                viz = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(viz)
                
                amplitude = self._get_amplitude(t)
                is_beat = any(abs(t - bt) < 0.1 for bt in self.beats)
                
                cx, cy = self.width // 2, self.height // 2
                base_r = 150
                num_bars = 60
                
                for i in range(num_bars):
                    angle = (2 * np.pi * i / num_bars) - np.pi / 2
                    np.random.seed(int(t * 100) + i)
                    bar_h = int(base_r * amplitude * 2 * (0.5 + 0.5 * np.random.random()))
                    bar_h = min(bar_h, 200)
                    if is_beat:
                        bar_h = int(bar_h * 1.5)
                    
                    inner_r = base_r
                    outer_r = base_r + bar_h
                    
                    x1 = int(cx + inner_r * np.cos(angle))
                    y1 = int(cy + inner_r * np.sin(angle))
                    x2 = int(cx + outer_r * np.cos(angle))
                    y2 = int(cy + outer_r * np.sin(angle))
                    
                    intensity = min(255, max(0, int(255 * amplitude)))
                    color = (intensity, 150, 255 - intensity, 200)
                    draw.line([x1, y1, x2, y2], fill=color, width=4)
                
                if is_beat:
                    glow_r = int(base_r * 0.7)
                    draw.ellipse([cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r], fill=(255, 255, 255, 150))
                
                return viz
            
            def _create_waveform(self, t):
                wave = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(wave)
                
                amplitude = self._get_amplitude(t)
                bar_count = 80
                bar_width = self.width // bar_count
                
                for i in range(bar_count):
                    np.random.seed(int(t * 50) + i)
                    factor = 0.3 + 0.7 * np.random.random()
                    bar_h = int((amplitude * 300) * factor)
                    bar_h = min(bar_h, self.height // 4)
                    
                    x = i * bar_width
                    y = self.height - bar_h - 40
                    
                    intensity = min(255, max(0, int(255 * amplitude)))
                    color = (255, intensity, 100, 220)
                    draw.rectangle([x + 2, y, x + bar_width - 2, self.height - 40], fill=color)
                
                return wave
        
        # Create generator instance
        gen = FrameGenerator(
            processed_image_bytes, audio_data, audio_sr, beat_times_list,
            change_times_list, total_duration, motion_type, transition_type,
            visualizer_style, width, height
        )
        
        # Free memory before render
        del processed_image_bytes
        gc.collect()
        
        # Render video
        output_filename = "studio_music_video.mp4"
        
        with st.spinner(f"🎬 Rendering {total_duration:.0f}s video at {fps}fps..."):
            video = VideoClip(lambda t: gen.get_frame(t), duration=total_duration)
            audio_clip = AudioFileClip(audio_path)
            final_video = video.set_audio(audio_clip)
            
            final_video.write_videofile(
                output_filename,
                fps=fps,
                codec="libx264",
                audio_codec="aac",
                preset='ultrafast',
                bitrate='3000k',
                threads=1,
                logger=None,
                write_logfile=False
            )
            
            # Close clips to free memory
            video.close()
            audio_clip.close()
            final_video.close()
            gc.collect()
        
        st.balloons()
        st.success("✨ Studio-Grade Video Ready!")
        
        # Read video into memory before displaying
        with open(output_filename, "rb") as f:
            video_bytes = f.read()
        
        st.video(video_bytes)
        
        st.download_button(
            "📥 Download Video (MP4)",
            data=video_bytes,
            file_name="Music_Video_Studio.mp4",
            mime="video/mp4",
            use_container_width=True
        )
        
        # Cleanup temp files
        try:
            os.remove(audio_path)
            os.remove(output_filename)
        except:
            pass
            
    except ImportError as e:
        st.error(f"❌ Missing library: {e}")
        st.code("pip install librosa moviepy pillow numpy")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
### 💡 Pro Tips:
- Use **high-resolution images** (1920x1080 or higher) for best quality
- Upload images in the **order** you want them to appear
- **Auto-Sync to Beats** creates dynamic transitions timed to music
- **Circular visualizer** looks best for center-focused images
- Rendering time depends on video length and quality (typically 1-3 minutes per minute of video)
""")
