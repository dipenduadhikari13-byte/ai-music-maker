import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Suno AI Architect",
    page_icon="🎹",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- 🔐 VIP LOCK SCREEN ---
# Load the password from Environment or Cloud Secrets
if "ACCESS_CODE" in st.secrets:
    correct_password = st.secrets["ACCESS_CODE"] # Cloud
else:
    correct_password = os.getenv("ACCESS_CODE")  # Local

# Initialize session state for login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Show Lock Screen if not logged in
if not st.session_state.authenticated:
    st.markdown("## 🔒 VIP Access Only")
    st.write("This tool is protected. Please enter your access code.")
    
    password_input = st.text_input("Enter Passcode:", type="password")
    
    if st.button("Unlock App"):
        if password_input == correct_password:
            st.session_state.authenticated = True
            st.success("Access Granted! Loading Studio...")
            st.rerun() # Reloads the app to show the content
        else:
            st.error("🚫 Incorrect Code. Please contact the admin.")
    
    st.stop() # 🛑 STOPS everything below this line until unlocked!

# --- 2. LOAD API KEY (Cloud & Local Support) ---
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Try to find the key in two places:
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]  # Works on Cloud
else:
    api_key = os.getenv("GOOGLE_API_KEY")   # Works on Laptop

# Verify
if not api_key:
    st.error("❌ API Key Not Found!")
    st.info("On Local: Check .env file. | On Cloud: Check 'Secrets' settings.")
    st.stop()
    # --- 3. SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Studio Settings")
    st.write("Current Model: **Gemini 2.5 Flash**")
    
    # Status Indicator
    if api_key:
        st.success("API Key Loaded! ✅")
    else:
        st.error("API Key Missing! ❌")
        st.info("Check your .env file.")
        st.stop() # Stop the app if no key

# --- 4. THE AI BRAIN (Function) ---
def generate_suno_prompt(topic, language, genre, vibe):
    client = genai.Client(api_key=api_key)
    
    # The "System Instruction" tells the AI how to behave
    system_instruction = (
        "You are an expert Suno AI Song Architect. "
        "Your goal is to generate inputs for Suno v3.5/v4.\n"
        "OUTPUT FORMAT:\n"
        "1. **Style Prompt:** A comma-separated string for the 'Style of Music' box.\n"
        "2. **Lyrics:** Complete lyrics with meta-tags [Verse], [Chorus], [Drop].\n"
        "RULES:\n"
        "- Use Romanized Script (English letters) for Hindi/Punjabi so Suno pronounces it right.\n"
        "- Add musical cues in brackets like (Heavy Bass, Ad-libs)."
    )

    user_prompt = (
        f"Create a hit song.\n"
        f"TOPIC: {topic}\n"
        f"LANGUAGE: {language}\n"
        f"GENRE: {genre}\n"
        f"SPECIFIC VIBE: {vibe}"
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=1.0, # High creativity
            )
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- 5. THE MAIN UI ---
st.title("🎹 Suno AI Song Architect")
st.markdown("Generate *perfect* inputs for Suno.ai in seconds.")
st.markdown("---")

# Layout: Two columns for inputs
col1, col2 = st.columns(2)

with col1:
    topic = st.text_input("📝 What is the song about?", placeholder="e.g. Village boy making it big in Canada")
    language = st.selectbox("🗣️ Language", ["Hindi", "Punjabi", "Haryanvi", "Bengali", "English"])

with col2:
    genre = st.selectbox("🎵 Genre", [
        "HipHop/Rap (Moosewala Style)", 
        "Soulful/Sad (Talwinder Style)", 
        "Desi Trap/Drill", 
        "Sufi Folk Fusion", 
        "Commercial Pop"
    ])
    vibe = st.text_input("✨ Specific Vibe/Instruments", placeholder="e.g. Flute, heavy 808, aggressive vocals")

# The Big Button
if st.button("🚀 Generate Song Assets", type="primary"):
    if not topic:
        st.warning("Please enter a song topic first!")
    else:
        with st.spinner("🎧 Cooking up the track... (Connecting to Gemini 2.5)"):
            result = generate_suno_prompt(topic, language, genre, vibe)
            
            # Show Result
            st.success("Generation Complete!")
            st.markdown("### 📋 Copy these into Suno:")
            
            # Text area allows easy copying
            st.text_area("Result", value=result, height=500)
            
            # Download Button
            st.download_button(
                label="💾 Download Lyrics (.txt)",
                data=result,
                file_name=f"suno_track_{topic.replace(' ', '_')}.txt",
                mime="text/plain"
            )