import streamlit as st
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- PAGE SETUP ---
st.set_page_config(page_title="Music Architect", page_icon="🎵", layout="wide")

# --- 1. SECURE API KEY SETUP ---
# Tries to find key in Streamlit Secrets (Web) OR .env (Local)
api_key = None
try:
    # Check Streamlit Cloud Secrets first
    api_key = st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback to local .env
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("❌ API Key Missing! Please add GOOGLE_API_KEY to your Secrets or .env file.")
    st.stop()

client = genai.Client(api_key=api_key)

# --- 2. SYSTEM INSTRUCTION ---
SYSTEM_INSTRUCTION = """
You are "The Music Architect," an expert song producer for Suno AI (v3.5/v4).
Your goal: Generate "Hit Song" lyrics and style tags.

RULES:
1. **Script:** Write lyrics in Romanized Script (English letters) for Hindi/Bengali/Punjabi.
2. **Structure:** Use strict tags: [Intro], [Verse 1], [Chorus], [Hook], [Drop], [Outro].
3. **Style:** Create a comma-separated style string (Genre, BPM, Vibe, Instruments).
4. **Format:** Output must be clean and ready to copy-paste.
"""

# --- 3. THE WEB INTERFACE ---
st.title("🎵 Music Architect: Suno Edition")
st.markdown("Design the blueprint for your next hit song.")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    topic = st.text_input("📝 What is the song about?", "A cyberpunk samurai fighting for love")
    language = st.selectbox("🌍 Language", [
        "English", "Hindi (Hinglish)", "Bengali (Banglish)", 
        "Punjabi (Romanized)", "Haryanvi (Desi)", "Urdu", "Spanish"
    ])

with col2:
    genre = st.selectbox("🎵 Genre / Vibe", [
        "HipHop / Rap (Aggressive)", "Trap / Drill (Dark)", 
        "Lo-Fi / Chill (Relaxing)", "EDM / House (Party)", 
        "Sufi / Folk (Soulful)", "Bollywood Commercial", "Heavy Metal"
    ])
    voice = st.selectbox("🎤 Voice Type", ["Male Vocals", "Female Vocals", "Duet", "Choir"])

# Generate Button
if st.button("🚀 Generate Blueprint"):
    with st.spinner("🎧 Connecting to the AI Matrix... Architecting your track..."):
        
        prompt = f"""
        Create a Suno AI song structure.
        TOPIC: {topic}
        LANGUAGE: {language}
        GENRE: {genre}
        VOCALS: {voice}
        
        Output Format:
        **STYLE PROMPT:** (The string for Suno)
        **LYRICS:** (Full lyrics with meta-tags)
        """
        
        # Failover Model List
        models = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]
        response_text = ""
        used_model = ""

        for model in models:
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.85
                    )
                )
                response_text = response.text
                used_model = model
                break # Stop if successful
            except Exception:
                continue # Try next model

        if response_text:
            st.success(f"✅ Generated using {used_model}")
            
            # Display Result
            st.subheader("Your Song Blueprint")
            st.text_area("Copy this output:", value=response_text, height=600)
            
            # Download Button
            st.download_button(
                label="💾 Download Lyrics (.txt)",
                data=response_text,
                file_name=f"Song_{int(time.time())}.txt",
                mime="text/plain"
            )
        else:
            st.error("❌ Failed to generate. All AI models are busy. Please try again.")