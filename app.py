import streamlit as st
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Music Architect Pro",
    page_icon="🎹",
    layout="wide"
)

# --- 1. SECURE API KEY & CLIENT SETUP ---
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

# --- 2. DYNAMIC MODEL FETCHING (The Fix) ---
@st.cache_resource(ttl=3600) # Cache this for 1 hour so we don't spam Google
def get_best_models():
    """
    Asks Google which models are currently available to your key.
    Sorts them by power: 2.0 Flash -> 1.5 Pro -> 1.5 Flash.
    """
    try:
        all_models = list(client.models.list())
        # Filter for models that can generate content (Gemini only)
        # We look for "generateContent" capability and "gemini" in the name
        valid_models = [
            m.name.replace("models/", "") 
            for m in all_models 
            if "gemini" in m.name and "vision" not in m.name
        ]
        
        # Custom Sorter: Prioritize the best models
        def model_priority(name):
            if "2.0-flash" in name: return 0      # 1st Priority (Newest/Fastest)
            if "1.5-pro" in name: return 1        # 2nd Priority (Smartest)
            if "1.5-flash" in name: return 2      # 3rd Priority (Reliable)
            return 3                              # Others

        valid_models.sort(key=model_priority)
        return valid_models
    except Exception as e:
        # Fallback list if the API fails to list models
        return ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]

# --- 3. THE "HIT MAKER" BRAIN (System Prompt v2.0) ---
SYSTEM_INSTRUCTION = """
You are "The Music Architect," a world-class producer for Suno AI (v3.5/v4).
Your goal is to engineer the perfect text prompt that results in a high-fidelity, structurally complex song.

### CRITICAL RULES:
1. **Script:** For Hindi/Bengali/Punjabi, you MUST use **Romanized Script** (English letters).
   - Bad: "मैं तुमसे प्यार करता हूँ" (Suno fails this).
   - Good: "Main tumse pyaar karta hoon" (Suno sings this perfectly).
2. **Meta-Tags:** Drive the song structure using strict tags:
   - [Intro], [Verse], [Chorus], [Bridge], [Hook], [Instrumental Solo], [Drop], [Outro].
   - Add performance cues: (whispered), (screamed), (autotune heavy), (choir backing).
3. **Style String:** Generate a focused style description (max 120 chars) for Suno's style box.
   - Format: Genre, BPM, Key Instruments, Vibe.
4. **Lyrics:** Create rhyme schemes (AABB, ABAB) that flow naturally. Use slang/ad-libs appropriate to the genre.

### OUTPUT FORMAT:
Provide the output in a clean format ready for copy-pasting.
"""

# --- 4. UI INTERFACE ---
st.title("🎹 Music Architect: Suno Edition")
st.markdown("### *Design Your Next Hit Song*")

col1, col2 = st.columns(2)

with col1:
    topic = st.text_input("📝 Song Concept", "A cyberpunk samurai fighting for love in Tokyo")
    language = st.selectbox("🌍 Language", [
        "English", "Hindi (Hinglish)", "Bengali (Banglish)", 
        "Punjabi (Romanized)", "Haryanvi (Desi)", "Urdu", "Spanish", "Japanese"
    ])

with col2:
    genre = st.selectbox("🎵 Genre / Vibe", [
        "HipHop / Rap (Aggressive)", "Trap / Drill (Dark 808s)", 
        "Lo-Fi / Chill (Study Beats)", "EDM / Progressive House", 
        "Sufi / Folk (Soulful)", "Bollywood Commercial (Party)", 
        "Heavy Metal / Rock", "Cinematic / Orchestral"
    ])
    voice = st.selectbox("🎤 Vocals", ["Male", "Female", "Duet", "Choir", "Instrumental Only"])

# --- 5. GENERATION LOGIC ---
if st.button("🚀 Architect Blueprint", type="primary"):
    
    # 1. Get the list of working models
    available_models = get_best_models()
    if not available_models:
        st.error("❌ Could not connect to Google AI. Check your API Key.")
        st.stop()
        
    status_box = st.empty()
    result_area = st.empty()
    
    prompt = f"""
    Create a detailed Suno AI song blueprint.
    TOPIC: {topic}
    LANGUAGE: {language}
    GENRE: {genre}
    VOCALS: {voice}
    
    Ensure the lyrics are catchy, rhythmic, and use Romanized script for non-English parts.
    Include a "Style Description" optimized for Suno v4.
    """
    
    success = False
    
    # 2. Try models one by one
    for model_name in available_models:
        try:
            status_box.info(f"🤖 Contacting **{model_name}**...")
            
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.9, # Higher creativity for music
                )
            )
            
            # If we get here, it worked!
            status_box.success(f"✅ Generated using **{model_name}**")
            result_area.text_area("Your Blueprint (Copy-Paste to Suno):", value=response.text, height=600)
            
            # Download Button
            st.download_button(
                label="💾 Download Lyrics (.txt)",
                data=response.text,
                file_name=f"Suno_Blueprint_{int(time.time())}.txt",
                mime="text/plain"
            )
            success = True
            break # Stop the loop
            
        except Exception as e:
            # Check for specific "Busy" errors
            error_str = str(e).lower()
            if "429" in error_str or "quota" in error_str:
                status_box.warning(f"⚠️ {model_name} is out of quota. Switching...")
            elif "503" in error_str or "overloaded" in error_str:
                status_box.warning(f"⚠️ {model_name} is overloaded. Switching...")
            else:
                # If it's a weird error, print it but keep trying
                print(f"Error with {model_name}: {e}")
                continue
    
    if not success:
        status_box.error("❌ All AI models are currently busy or down. Please try again in 2 minutes.")