import streamlit as st
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import tempfile
from io import BytesIO

# --- CONFIGURATION ---
st.set_page_config(page_title="Music Video Maker", page_icon="🎵", layout="wide")

# Video Settings
WIDTH = 1280
HEIGHT = 720
FPS = 24

# --- HELPER FUNCTIONS ---

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

# --- UI LAYOUT ---
st.title("🎵 Music Video Generator")
st.markdown("Upload your song and image to create a professional music video")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Files")
    
    audio_file = st.file_uploader("Upload Audio (MP3/WAV)", type=["mp3", "wav", "m4a"])
    image_file = st.file_uploader("Upload Background Image", type=["jpg", "jpeg", "png"])
    
    st.subheader("🎨 Customization")
    
    effect_type = st.selectbox(
        "Video Effect",
        ["Slow Zoom In", "Slow Zoom Out", "Ken Burns (Pan & Zoom)", "Pulse Effect", "Static"]
    )
    
    add_text = st.checkbox("Add Text Overlay")
    
    if add_text:
        overlay_text = st.text_input("Text", placeholder="Song Title or Artist Name")
        text_color = st.color_picker("Text Color", "#FFFFFF")
    
    enhance_colors = st.checkbox("Enhance Colors", value=True)
    
    render_button = st.button("🎬 Generate Video", type="primary", use_container_width=True)

with col2:
    st.subheader("👁️ Preview")
    if image_file:
        preview_img = Image.open(image_file)
        st.image(preview_img, caption="Your Background Image", use_column_width=True)
    else:
        st.info("Upload an image to see preview")

# --- VIDEO GENERATION ---
if render_button:
    if not audio_file:
        st.error("❌ Please upload an audio file!")
        st.stop()
    
    if not image_file:
        st.error("❌ Please upload an image file!")
        st.stop()
    
    try:
        # Import heavy libraries only when needed
        import librosa
        from moviepy.editor import VideoClip, AudioFileClip, CompositeVideoClip, TextClip
        
        with st.spinner("🎧 Processing audio..."):
            # Save audio to temp file
            audio_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            audio_temp.write(audio_file.read())
            audio_temp.close()
            audio_path = audio_temp.name
            
            # Load audio and get duration
            y, sr = librosa.load(audio_path, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)
            
            st.success(f"✅ Audio loaded: {duration:.1f} seconds")
        
        with st.spinner("🖼️ Processing image..."):
            # Load and prepare image
            base_image = Image.open(image_file)
            base_image = resize_image_to_fit(base_image, WIDTH, HEIGHT)
            
            # Enhance if requested
            if enhance_colors:
                base_image = enhance_image(base_image)
            
            # Add text if requested
            if add_text and overlay_text:
                text_position = (50, HEIGHT - 150)
                # Convert hex to RGB
                text_rgb = tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
                base_image = add_overlay_text(base_image, overlay_text, text_position, color=text_rgb)
            
            st.success("✅ Image processed")
        
        with st.spinner("🎬 Rendering video... This may take a few minutes"):
            # Frame generation function
            def make_frame(t):
                # Calculate progress (0 to 1)
                progress = t / duration
                
                # Create a copy of the base image
                frame = base_image.copy()
                
                # Apply selected effect
                if effect_type == "Slow Zoom In":
                    zoom = 1.0 + (progress * 0.2)  # Zoom from 1.0 to 1.2
                    frame = apply_zoom_effect(frame, zoom)
                
                elif effect_type == "Slow Zoom Out":
                    zoom = 1.2 - (progress * 0.2)  # Zoom from 1.2 to 1.0
                    frame = apply_zoom_effect(frame, zoom)
                
                elif effect_type == "Ken Burns (Pan & Zoom)":
                    # Combine zoom and slight pan
                    zoom = 1.0 + (0.1 * np.sin(2 * np.pi * progress))
                    frame = apply_zoom_effect(frame, zoom)
                
                elif effect_type == "Pulse Effect":
                    # Pulsing zoom effect synced to beat
                    pulse_freq = 0.5  # Pulses per second
                    zoom = 1.0 + (0.05 * np.sin(2 * np.pi * pulse_freq * t))
                    frame = apply_zoom_effect(frame, zoom)
                
                # Static = no effect, just return the frame
                
                # Convert to numpy array
                return np.array(frame)
            
            # Create video clip
            video_clip = VideoClip(make_frame, duration=duration)
            
            # Add audio
            audio_clip = AudioFileClip(audio_path)
            final_video = video_clip.set_audio(audio_clip)
            
            # Output path
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def progress_callback(current_frame, total_frames):
                progress = current_frame / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Rendering: {int(progress * 100)}% ({current_frame}/{total_frames} frames)")
            
            # Render video
            final_video.write_videofile(
                output_path,
                fps=FPS,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=tempfile.mktemp(suffix='.m4a'),
                remove_temp=True,
                preset='medium',
                threads=4,
                logger=None
            )
            
            progress_bar.empty()
            status_text.empty()
        
        st.balloons()
        st.success("✨ Video generated successfully!")
        
        # Display video
        st.video(output_path)
        
        # Download button
        with open(output_path, 'rb') as video_file:
            video_bytes = video_file.read()
            st.download_button(
                label="📥 Download Video",
                data=video_bytes,
                file_name="music_video.mp4",
                mime="video/mp4",
                use_container_width=True
            )
        
        # Cleanup
        try:
            os.unlink(audio_path)
            os.unlink(output_path)
        except:
            pass
            
    except ImportError as e:
        st.error(f"❌ Missing required library: {e}")
        st.info("Install requirements: `pip install librosa moviepy pillow streamlit`")
    
    except Exception as e:
        st.error(f"❌ Error generating video: {str(e)}")
        st.exception(e)

# --- FOOTER ---
st.markdown("---")
st.caption("💡 Tip: Use high-resolution images (1920x1080 or larger) for best results")