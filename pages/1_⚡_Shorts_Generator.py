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
    
    analysis_mode = st.selectbox(
        "Analysis Mode",
        [
            "🔥 Maximum Energy (Loudest/Most Intense)",
            "🎵 Beat-Aligned (Starts on Strong Beat)",
            "🎭 Dynamic Range (Most Variation)",
            "🎬 Balanced (All Factors)"
        ]
    )
    
    add_effects = st.checkbox("Add Viral Effects", value=False, help="Add zoom, flash effects at peaks")
    
    manual_override = st.checkbox("Manual Selection", value=False)
    
    if manual_override and uploaded_video:
        start_time = st.number_input("Start Time (seconds)", 0.0, duration - 30, 0.0, 1.0)
        end_time = st.number_input("End Time (seconds)", start_time + 1, duration, min(start_time + 30, duration), 1.0)

# --- PROCESSING ---
if uploaded_video and st.button("🎯 Find Viral Hook", type="primary", use_container_width=True):
    with st.spinner("🔍 Analyzing video for viral potential..."):
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Extract audio
        status_text.text("📊 Analyzing audio energy...")
        progress_bar.progress(20)
        
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
        
        progress_bar.progress(40)
        status_text.text("🎵 Detecting beats and rhythm...")
        
        # Step 3: Detect beats
        beat_times, tempo = detect_beats(y, sr)
        
        progress_bar.progress(60)
        status_text.text("🎯 Finding optimal 30s segment...")
        
        # Step 4: Find best segment
        if manual_override:
            best_start = start_time
            best_end = end_time
        else:
            best_start, best_end = find_best_30s_segment(
                energy_score, sr, duration, beat_times, y
            )
        
        progress_bar.progress(80)
        status_text.text("✂️ Extracting and optimizing clip...")
        
        # Step 5: Extract clip
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix="_short.mp4").name
        success = extract_clip(input_path, best_start, best_end, output_path)
        
        if not success:
            st.error("Failed to create short. Please try again.")
            st.stop()
        
        progress_bar.progress(100)
        status_text.text("✅ Viral hook created!")
        
        # Display results
        st.success("🎉 Your Viral Short is Ready!")
        
        col_result1, col_result2 = st.columns([2, 1])
        
        with col_result1:
            st.subheader("📱 Preview (9:16 Format)")
            st.video(output_path)
        
        with col_result2:
            st.subheader("📊 Analysis")
            st.metric("Start Time", f"{best_start:.1f}s")
            st.metric("End Time", f"{best_end:.1f}s")
            st.metric("Duration", f"{best_end - best_start:.1f}s")
            
            if tempo:
                st.metric("BPM", f"{tempo:.0f}")
            
            # Energy visualization
            st.subheader("🔥 Energy Graph")
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(6, 3))
            time_axis = np.linspace(0, duration, len(energy_score))
            ax.plot(time_axis, energy_score, color='#FF4B4B', linewidth=2)
            ax.axvspan(best_start, best_end, alpha=0.3, color='green', label='Selected Clip')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Energy')
            ax.set_title('Audio Energy Analysis')
            ax.legend()
            st.pyplot(fig)
        
        # Download button
        with open(output_path, 'rb') as f:
            st.download_button(
                label="📥 Download Shorts Video",
                data=f,
                file_name=f"viral_short_{int(best_start)}s.mp4",
                mime="video/mp4",
                use_container_width=True,
                type="primary"
            )
        
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
    <p>💡 <b>Pro Tip:</b> The algorithm finds segments with high energy, dynamic range, and beat alignment for maximum viral potential!</p>
</div>
""", unsafe_allow_html=True)
