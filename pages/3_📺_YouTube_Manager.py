import streamlit as st
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube Manager",
    page_icon="📺",
    layout="wide"
)

# --- 1. SECURE API KEY SETUP ---
api_key = None
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ API Key Missing! Please add GOOGLE_API_KEY to secrets.")
    st.stop()

client = genai.Client(api_key=api_key)

# --- 2. SYSTEM BRAIN (YouTube SEO Expert) ---
SYSTEM_INSTRUCTION = """
You are a "YouTube Growth Expert" for a music channel named 'Soul Note Originals'.
Your goal is to write high-CTR (Click Through Rate) Titles, engaging Descriptions, and viral Tags.

RULES:
1. **Titles:** Generate 3 options. One Emotional, one Clickbaity (but honest), one Search-Optimized.
2. **Description:** - Start with a strong hook.
   - Include a "Story behind the track" section (improvise based on the topic).
   - End with a Call to Action (Subscribe to Soul Note Originals).
3. **Tags:** Provide a comma-separated list of high-volume keywords.
4. **Hashtags:** Provide 3-5 relevant hashtags.
"""

# --- 3. UI INTERFACE ---
st.title("📺 YouTube Manager")
st.markdown("### *Metadata Generator for Soul Note Originals*")

# Inputs
col1, col2 = st.columns(2)

with col1:
    video_topic = st.text_input("📝 Video Topic / Song Name", "Cyberpunk Samurai Lo-Fi")
    music_genre = st.selectbox("🎵 Genre", ["Lo-Fi", "Trap", "Classical", "Cinematic", "Ambient", "Phonk", "Drill"])

with col2:
    mood = st.selectbox("✨ Mood", ["Relaxing", "Dark/Aggressive", "Motivational", "Sad/Emotional", "Party/Upbeat"])
    channel_name = st.text_input("📢 Channel Name", "Soul Note Originals")

# Generate Button
if st.button("🚀 Generate Metadata", type="primary"):
    
    with st.spinner("🧠 Analyzing Algorithm... Writing Metadata..."):
        prompt = f"""
        Generate YouTube Video Metadata.
        TOPIC: {video_topic}
        GENRE: {music_genre}
        MOOD: {mood}
        CHANNEL: {channel_name}
        
        OUTPUT FORMAT:
        **TITLES** (3 Options)
        **DESCRIPTION** (Full text with Chapters placeholder)
        **TAGS** (Comma separated)
        **HASHTAGS**
        """
        
        # Failover Logic
        models = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]
        response_text = ""
        
        for model in models:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.8
                    )
                )
                response_text = response.text
                break
            except Exception:
                continue

        if response_text:
            st.success("✅ Metadata Generated!")
            st.subheader("📋 Copy to YouTube Studio")
            st.text_area("Full Output", value=response_text, height=600)
        else:
            st.error("❌ AI Busy. Try again in 1 minute.")