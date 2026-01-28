import os
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 1. CONFIGURATION ---
# Load API Key
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ API Key missing. Please check your .env file.")
    exit()

client = genai.Client(api_key=api_key)

# Failover Model List: Tries the smartest/newest first
MODEL_LIST = [
    "gemini-2.0-flash-exp", 
    "gemini-1.5-pro", 
    "gemini-1.5-flash"
]

# --- 2. THE GENIUS ARCHITECT SYSTEM PROMPT ---
# This is the "brain" that teaches Gemini how to control Suno AI
SYSTEM_INSTRUCTION = """
You are "The Music Architect," an expert song producer specializing in prompting Suno AI (v3.5/v4).
Your goal is to generate lyrics and style tags that create "Hit Songs" with perfect flow, rhythm, and structure.

### CRITICAL RULES FOR SUNO OPTIMIZATION:
1. **Script:** ALWAYS write lyrics in **Romanized Script** (English letters) for non-English languages (Hindi, Bengali, Punjabi, etc.). This ensures Suno pronounces the words correctly.
2. **Structure:** You MUST use strict Meta Tags to control the song structure:
   - [Intro], [Verse 1], [Chorus], [Verse 2], [Bridge], [Guitar Solo], [Drop], [Outro].
   - Use style cues inside lyrics like (whisper), (shout), (echo).
3. **Style Box:** Generate a specific, comma-separated string for Suno's "Style of Music" box. Include Genre, BPM, Vibe, and Instruments.
4. **Rhyme & Flow:** The lyrics must have a strong rhyme scheme (AABB or ABAB). Avoid generic AI poetry. Use slang and cultural references appropriate to the language.

### OUTPUT FORMAT (Strict JSON-like structure):
---
**STYLE PROMPT:** (Copy this into Suno's Style Box)
[Genre], [BPM], [Vibe], [Male/Female Vocals], [Key Instruments]

**LYRICS:** (Copy this into Suno's Lyrics Box)
[Intro]
...
[Verse 1]
...
[Chorus]
...
---
"""

def get_user_choice(options, prompt_text):
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    
    while True:
        choice = input("  👉 Select (Number): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("  ❌ Invalid choice. Try again.")

def get_inputs():
    print("\n🎹 --- MUSIC ARCHITECT: SUNO EDITION --- 🎹")
    print("Design the blueprint for your next hit song.\n")

    # 1. Topic
    topic = input("📝 What is the song about? (e.g., 'A hacker falling in love', 'Village life'): ")

    # 2. Language
    languages = [
        "English", "Hindi (Hinglish)", "Bengali (Banglish)", 
        "Punjabi (Romanized)", "Haryanvi (Desi)", "Urdu", 
        "Tamil", "Telugu", "Spanish", "Mix (Hindi + English)"
    ]
    lang = get_user_choice(languages, "🌍 Select Language:")

    # 3. Vibe / Genre
    genres = [
        "HipHop / Rap (Aggressive)", "Trap / Drill (Dark)", 
        "Lo-Fi / Chill (Relaxing)", "EDM / House (Party)", 
        "Sufi / Folk (Soulful)", "Romantic / Acoustic (Soft)",
        "Phonk (High Energy)", "Heavy Metal / Rock",
        "Synthwave (Retro 80s)", "Bollywood Commercial"
    ]
    genre = get_user_choice(genres, "🎵 Select Genre/Style:")

    # 4. Voice Gender
    voices = ["Male Vocals", "Female Vocals", "Duet (Male & Female)", "Choir/Group"]
    voice = get_user_choice(voices, "🎤 Select Voice Type:")

    return topic, lang, genre, voice

def generate_song_blueprint(topic, language, genre, voice):
    # Dynamic Prompt Construction
    prompt = f"""
    Create a high-quality song structure for Suno AI.
    
    DETAILS:
    - **Topic:** {topic}
    - **Language:** {language} (Use Romanized English script for correct pronunciation)
    - **Genre:** {genre}
    - **Vocals:** {voice}
    
    REQUIREMENTS:
    1. Create a "Style Description" that guarantees the genre vibe (mention BPM, specific instruments like 808s, Sitar, Guitar, etc.).
    2. Write the **Lyrics** with high energy or deep emotion (depending on genre).
    3. Use Metatags like [Bass Drop], [Beat Switch], [Melodic Interlude] to make it interesting.
    4. If Rap/Drill: Use ad-libs (Skrrt, Yeah, Brrr).
    5. If Soulful: Use poetic metaphors.
    """

    print(f"\n🎧 Architecting track... (Connecting to AI Matrix)")

    for model_name in MODEL_LIST:
        try:
            # print(f"   Trying {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.85, # High creativity but structured
                )
            )
            return response.text, model_name

        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "quota" in error_msg:
                print(f"   ⚠️ {model_name} busy. Switching brain...")
            elif "404" in error_msg:
                continue
            else:
                print(f"   ⚠️ Error: {e}")
                
    return None, None

# --- MAIN LOOP ---
if __name__ == "__main__":
    while True:
        topic, lang, genre, voice = get_inputs()
        
        result, model_used = generate_song_blueprint(topic, lang, genre, voice)
        
        if result:
            print("\n" + "="*50)
            print(f"🤖 BLUEPRINT GENERATED BY: {model_used}")
            print("="*50)
            print(result)
            print("="*50)
            
            # Save to file
            filename = f"Song_{topic[:15].replace(' ', '_')}_{int(time.time())}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(result)
            
            print(f"\n💾 Saved to: {filename}")
            print("👉 Copy 'STYLE PROMPT' to Suno's Style Box.")
            print("👉 Copy 'LYRICS' to Suno's Lyrics Box.")
            print("👉 Enable 'Custom Mode' in Suno to use this.")
        else:
            print("❌ Failed to generate. Please check your Internet or API Key.")

        # Re-run option
        if input("\n🔄 Create another song? (y/n): ").lower() != 'y':
            print("👋 Keep making hits!")
            break