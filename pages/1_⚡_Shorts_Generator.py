import streamlit as st
import os
import tempfile
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
import librosa
from PIL import Image
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Shorts Generator",
    page_icon="⚡",
    layout="wide"
)

# --- HELPER FUNCTIONS ---

def analyze_audio_energy(audio_path, sr=22050):
    """
    Analyze audio to find segments with highest energy/intensity.
    Returns timestamps of peak energy windows.
    """
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Calculate spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Normalize both features
        rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-8)
        centroid_normalized = (spectral_centroid - np.min(spectral_centroid)) / (np.max(spectral_centroid) - np.min(spectral_centroid) + 1e-8)
        
        # Combined energy score (60% energy, 40% brightness)
        energy_score = 0.6 * rms_normalized + 0.4 * centroid_normalized
        
        return energy_score, sr, y
    except Exception as e:
        st.error(f"Audio analysis failed: {e}")
        return None, None, None

def detect_beats(y, sr):
    """Detect beat locations in audio."""
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        return beat_times, tempo
    except Exception as e:
        st.warning(f"Beat detection failed: {e}")
        return None, None

def find_best_30s_segment(energy_score, sr, duration, beat_times=None, y=None):
    """
    Find the best 30-second window based on:
    1. Maximum average energy
    2. Beat alignment (start on a beat)
    3. Dynamic range (variation in the segment)
    """
    window_duration = 30  # seconds
    hop_size = 512  # librosa default
    
    # Convert energy score to time-based
    frame_duration = hop_size / sr
    total_frames = len(energy_score)
    
    # Calculate window size in frames
    window_frames = int(window_duration / frame_duration)
    
    if window_frames > total_frames:
        st.warning("Video is shorter than 30 seconds. Using entire video.")
        return 0, min(duration, 30)
    
    best_score = -1
    best_start_time = 0
    
    # Slide window through the audio
    for i in range(0, total_frames - window_frames, window_frames // 10):  # 10% overlap
        window_energy = energy_score[i:i+window_frames]
        
        # Calculate metrics
        avg_energy = np.mean(window_energy)
        energy_variance = np.var(window_energy)  # Higher variance = more dynamic
        peak_count = np.sum(window_energy > np.percentile(window_energy, 80))  # Count of high-energy moments
        
        # Combined score
        score = (avg_energy * 0.5) + (energy_variance * 0.3) + (peak_count / len(window_energy) * 0.2)
        
        if score > best_score:
            best_score = score
            best_start_time = i * frame_duration
    
    # Align to nearest beat if available
    if beat_times is not None and len(beat_times) > 0:
        # Find closest beat to start time
        closest_beat_idx = np.argmin(np.abs(beat_times - best_start_time))
        aligned_start = beat_times[closest_beat_idx]
        
        # Make sure we don't exceed video duration
        if aligned_start + window_duration <= duration:
            best_start_time = aligned_start
    
    end_time = min(best_start_time + window_duration, duration)
    
    return best_start_time, end_time

def extract_clip(video_path, start_time, end_time, output_path):
    """Extract a segment from video and resize for Shorts (9:16 aspect ratio)."""
    try:
        video = VideoFileClip(video_path)
        
        # Extract the segment
        clip = video.subclip(start_time, end_time)
        
        # Get original dimensions
        w, h = clip.size
        target_aspect = 9 / 16  # Shorts aspect ratio
        current_aspect = w / h
        
        # Crop to 9:16 if needed
        if current_aspect > target_aspect:
            # Video is too wide, crop sides
            new_width = int(h * target_aspect)
            x_center = w / 2
            x1 = int(x_center - new_width / 2)
            clip = clip.crop(x1=x1, width=new_width)
        else:
            # Video is too tall, crop top/bottom
            new_height = int(w / target_aspect)
            y_center = h / 2
            y1 = int(y_center - new_height / 2)
            clip = clip.crop(y1=y1, height=new_height)
        
        # Resize to standard Shorts resolution (1080x1920)
        clip = clip.resize((1080, 1920))
        
        # Write output
        clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=30,
            preset='ultrafast',
            logger=None
        )
        
        clip.close()
        video.close()
        
        return True
    except Exception as e:
        st.error(f"Video extraction failed: {e}")
        return False

def get_video_thumbnail(video_path, time=5):
    """Get a thumbnail from the video at specified time."""
    try:
        video = VideoFileClip(video_path)
        frame = video.get_frame(min(time, video.duration - 1))
        video.close()
        return Image.fromarray(frame.astype('uint8'), 'RGB')
    except Exception as e:
        st.warning(f"Could not generate thumbnail: {e}")
        return None

def find_multiple_hooks(energy_score, sr, duration, beat_times, num_clips=3, clip_duration=30):
    """Find multiple viral hooks from the video."""
    hop_size = 512
    frame_duration = hop_size / sr
    total_frames = len(energy_score)
    window_frames = int(clip_duration / frame_duration)
    
    if window_frames > total_frames:
        return [(0, min(duration, clip_duration))]
    
    # Find all potential segments with scores
    segments = []
    for i in range(0, total_frames - window_frames, window_frames // 5):
        window_energy = energy_score[i:i+window_frames]
        avg_energy = np.mean(window_energy)
        energy_variance = np.var(window_energy)
        peak_count = np.sum(window_energy > np.percentile(window_energy, 80))
        score = (avg_energy * 0.5) + (energy_variance * 0.3) + (peak_count / len(window_energy) * 0.2)
        
        start_time = i * frame_duration
        # Align to beat
        if beat_times is not None and len(beat_times) > 0:
            closest_beat_idx = np.argmin(np.abs(beat_times - start_time))
            start_time = beat_times[closest_beat_idx]
        
        end_time = min(start_time + clip_duration, duration)
        if end_time - start_time >= clip_duration * 0.8:  # At least 80% of desired length
            segments.append((start_time, end_time, score))
    
    # Sort by score and remove overlapping segments
    segments.sort(key=lambda x: x[2], reverse=True)
    selected = []
    for seg in segments:
        if len(selected) >= num_clips:
            break
        # Check if overlaps with already selected
        overlap = False
        for sel in selected:
            if not (seg[1] <= sel[0] or seg[0] >= sel[1]):
                overlap = True
                break
        if not overlap:
            selected.append((seg[0], seg[1]))
    
    return selected if selected else [(0, min(duration, clip_duration))]

def add_caption_overlay(clip, text, position='bottom', font_size=70, style='bold'):
    """Add trending-style captions to video."""
    from moviepy.editor import TextClip, CompositeVideoClip
    
    try:
        # Calculate position
        w, h = clip.size
        if position == 'bottom':
            y_pos = h - 200
        elif position == 'top':
            y_pos = 100
        else:
            y_pos = h // 2
        
        # Create caption with trendy styling
        txt_clip = TextClip(
            text,
            fontsize=font_size,
            color='white',
            font='Arial-Bold' if style == 'bold' else 'Arial',
            stroke_color='black',
            stroke_width=3,
            method='caption',
            size=(w - 100, None),
            align='center'
        ).set_position(('center', y_pos)).set_duration(clip.duration)
        
        return CompositeVideoClip([clip, txt_clip])
    except:
        # If TextClip fails, return original
        return clip

def add_viral_effects(clip, beat_times, effect_type='zoom'):
    """Add viral editing effects synced to beats."""
    from moviepy.editor import CompositeVideoClip
    
    if effect_type == 'zoom':
        # Add zoom effect on beats
        def zoom_effect(get_frame, t):
            frame = get_frame(t)
            # Check if near a beat
            is_beat = any(abs(t - bt) < 0.1 for bt in beat_times if abs(t - bt) < 0.3)
            if is_beat:
                # Slight zoom in
                h, w = frame.shape[:2]
                zoom_factor = 1.1
                new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
                top = (h - new_h) // 2
                left = (w - new_w) // 2
                cropped = frame[top:top+new_h, left:left+new_w]
                from PIL import Image
                img = Image.fromarray(cropped)
                img = img.resize((w, h), Image.Resampling.LANCZOS)
                return np.array(img)
            return frame
        
        return clip.fl(zoom_effect)
    
    return clip

def smart_crop_to_shorts(clip, focus='center'):
    """Intelligently crop to 9:16 with focus on action."""
    w, h = clip.size
    target_aspect = 9 / 16
    current_aspect = w / h
    
    if current_aspect > target_aspect:
        # Video is too wide - need to crop sides
        new_width = int(h * target_aspect)
        
        if focus == 'center':
            x1 = int((w - new_width) / 2)
        elif focus == 'left':
            x1 = 0
        elif focus == 'right':
            x1 = w - new_width
        else:
            x1 = int((w - new_width) / 2)
        
        clip = clip.crop(x1=x1, width=new_width)
    else:
        # Video is too tall - crop top/bottom
        new_height = int(w / target_aspect)
        y1 = int((h - new_height) / 2)
        clip = clip.crop(y1=y1, height=new_height)
    
    # Resize to standard Shorts resolution
    return clip.resize((1080, 1920))

# --- UI LAYOUT ---
st.title("⚡ Viral Shorts Generator")
st.markdown("Upload a music video and we'll find the most engaging 30-second hook!")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Upload Video")
    uploaded_video = st.file_uploader(
        "Upload Music Video (MP4, MOV, AVI)",
        type=["mp4", "mov", "avi", "mkv"],
        help="Upload your full-length music video"
    )
    
    if uploaded_video:
        # Save uploaded file
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_video.read())
        input_path = temp_input.name
        temp_input.close()
        
        st.video(input_path)
        
        # Get video info
        try:
            video = VideoFileClip(input_path)
            duration = video.duration
            fps = video.fps
            resolution = f"{video.w}x{video.h}"
            video.close()
            
            st.info(f"📊 **Duration:** {duration:.1f}s | **FPS:** {fps} | **Resolution:** {resolution}")
        except Exception as e:
            st.error(f"Could not read video: {e}")
            st.stop()

with col2:
    st.subheader("⚙️ Settings")
    
    num_clips = st.selectbox(
        "Number of Shorts",
        [1, 2, 3],
        index=2,
        help="Generate multiple shorts from different parts of the video"
    )
    
    clip_duration = st.slider(
        "Clip Duration (seconds)",
        15, 60, 30,
        help="Length of each short (30s recommended for YouTube/IG)"
    )
    
    st.divider()
    
    analysis_mode = st.selectbox(
        "Analysis Mode",
        [
            "🎬 Balanced (Recommended)",
            "🔥 Maximum Energy",
            "🎵 Beat-Aligned",
            "🎭 Dynamic Range"
        ]
    )
    
    st.divider()
    st.subheader("✨ Viral Elements")
    
    add_captions = st.checkbox("Auto Captions", value=True, help="Add trending-style text overlays")
    if add_captions:
        caption_text = st.text_input("Caption Text", "Wait for it... 🔥", help="Hook viewers in first 3 seconds")
    
    add_effects = st.checkbox("Beat Sync Effects", value=True, help="Zoom/flash on beats")
    
    crop_focus = st.selectbox(
        "Crop Focus",
        ["Center", "Left", "Right"],
        help="Where to focus when cropping to 9:16"
    )
    
    st.divider()
    manual_override = st.checkbox("Manual Selection", value=False)
    
    if manual_override and uploaded_video:
        start_time = st.number_input("Start Time (seconds)", 0.0, duration - clip_duration, 0.0, 1.0)
        end_time = st.number_input("End Time (seconds)", start_time + 1, duration, min(start_time + clip_duration, duration), 1.0)

# --- PROCESSING ---
if uploaded_video and st.button("🎯 Generate Viral Shorts", type="primary", use_container_width=True):
    with st.spinner("🔍 Analyzing video for viral potential..."):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Extract audio
        status_text.text("📊 Analyzing audio energy...")
        progress_bar.progress(15)
        
        try:
            video_clip = VideoFileClip(input_path)
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            video_clip.audio.write_audiofile(temp_audio.name, logger=None)
            audio_path = temp_audio.name
            video_clip.close()
        except Exception as e:
            st.error(f"Audio extraction failed: {e}")
            st.stop()
        
        # Step 2: Analyze audio
        energy_score, sr, y = analyze_audio_energy(audio_path)
        if energy_score is None:
            st.error("Failed to analyze audio. Please try a different video.")
            st.stop()
        
        progress_bar.progress(30)
        status_text.text("🎵 Detecting beats and rhythm...")
        
        # Step 3: Detect beats
        beat_times, tempo = detect_beats(y, sr)
        
        progress_bar.progress(45)
        status_text.text("🎯 Finding optimal segments...")
        
        # Step 4: Find best segments
        if manual_override:
            segments = [(start_time, end_time)]
        else:
            segments = find_multiple_hooks(
                energy_score, sr, duration, beat_times, 
                num_clips=num_clips, clip_duration=clip_duration
            )
        
        progress_bar.progress(60)
        status_text.text(f"✂️ Creating {len(segments)} viral short(s)...")
        
        # Step 5: Extract and process clips
        output_paths = []
        for idx, (start, end) in enumerate(segments):
            try:
                # Load video
                video = VideoFileClip(input_path)
                clip = video.subclip(start, end)
                
                # Smart crop to 9:16
                clip = smart_crop_to_shorts(clip, focus=crop_focus.lower())
                
                # Add viral effects if enabled
                if add_effects and beat_times is not None:
                    clip_beat_times = [bt - start for bt in beat_times if start <= bt <= end]
                    clip = add_viral_effects(clip, clip_beat_times, 'zoom')
                
                # Add captions if enabled
                if add_captions and caption_text:
                    clip = add_caption_overlay(clip, caption_text, position='bottom')
                
                # Save clip
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix=f"_short_{idx+1}.mp4").name
                clip.write_videofile(
                    output_path,
                    codec='libx264',
                    audio_codec='aac',
                    fps=30,
                    preset='medium',
                    logger=None
                )
                
                output_paths.append((output_path, start, end))
                clip.close()
                video.close()
                
                progress_bar.progress(60 + int(30 * (idx + 1) / len(segments)))
            except Exception as e:
                st.warning(f"Failed to create clip {idx+1}: {e}")
                continue
        
        if not output_paths:
            st.error("Failed to create any shorts. Please try again.")
            st.stop()
        
        progress_bar.progress(100)
        status_text.text("✅ Viral shorts created!")
        
        # Display results
        st.success(f"🎉 {len(output_paths)} Viral Short(s) Ready!")
        
        # Show each clip
        for idx, (path, start, end) in enumerate(output_paths):
            st.divider()
            st.subheader(f"📱 Short #{idx+1}")
            
            col_r1, col_r2 = st.columns([2, 1])
            
            with col_r1:
                st.video(path)
            
            with col_r2:
                st.metric("Start Time", f"{start:.1f}s")
                st.metric("End Time", f"{end:.1f}s")
                st.metric("Duration", f"{end - start:.1f}s")
                
                if tempo:
                    st.metric("BPM", f"{tempo:.0f}")
                
                with open(path, 'rb') as f:
                    st.download_button(
                        label=f"📥 Download Short #{idx+1}",
                        data=f,
                        file_name=f"viral_short_{idx+1}_{int(start)}s.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                        key=f"download_{idx}"
                    )
        
        # Energy visualization
        st.divider()
        st.subheader("🔥 Energy Analysis")
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 4))
        time_axis = np.linspace(0, duration, len(energy_score))
        ax.plot(time_axis, energy_score, color='#FF4B4B', linewidth=2, alpha=0.7)
        
        # Highlight selected segments
        colors = ['green', 'blue', 'orange']
        for idx, (_, start, end) in enumerate(output_paths):
            ax.axvspan(start, end, alpha=0.3, color=colors[idx % 3], label=f'Short #{idx+1}')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Energy', fontsize=12)
        ax.set_title('Audio Energy Analysis - Selected Viral Hooks', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Cleanup
        try:
            os.unlink(input_path)
            os.unlink(audio_path)
        except:
            pass

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💡 <b>Pro Tips:</b></p>
    <p>• Generate 3 shorts for A/B testing on different platforms</p>
    <p>• Add captions for 85% higher engagement (most watch on mute)</p>
    <p>• Beat-synced effects boost retention by keeping viewers hooked</p>
    <p>• 30s clips perform best on YouTube Shorts, 15-20s for Instagram Reels</p>
</div>
""", unsafe_allow_html=True)
