import streamlit as st
import os
import time
import re
import traceback
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
You are "The Music Architect," a world-class Suno AI producer crafting genius-level lyrics.
Your mission: Create emotionally compelling, structurally perfect songs optimized for Suno v4.

### LYRIC EXCELLENCE RULES:

**1. LANGUAGE & SCRIPT:**
- Non-English: Use ONLY Romanized Script (English letters).
  - Hindi: "Main tumse pyaar karta hoon" ✓
  - Bengali: "Ami tomake bhalobashi" ✓
  - Punjabi: "Main tenu chaunda haan" ✓
  - Haryanvi: "Main tenu chahta hoon" ✓
- NEVER use Devanagari, Bengali script, or Arabic characters.
- IMPORTANT: Break multi-syllable words into phonetic chunks for Suno clarity.
  - "pyaar" → keep as is (2 syllables: py-aar)
  - "samjhauta" → "sam-jhau-ta" (3 syllables, spread across beats)
  - "akela" → "a-ke-la" (3 syllables, clear separation)

**2. LYRIC STRUCTURE (Apply STRICT Format):**
[Intro] - 1-2 lines, atmospheric hook
[Verse 1] - 8-12 lines, story setup, rhyme scheme AABB or ABAB
[Chorus] - 4-6 lines, CATCHY & REPETITIVE, singable hook
[Verse 2] - 8-12 lines, story development, NEW rhyme scheme
[Chorus] - Repeat with variations
[Bridge] - 4-8 lines, emotional peak or plot twist
[Chorus] - Final powerful rendition
[Outro] - 1-4 lines, resolution/fade

**3. LYRIC QUALITY STANDARDS:**
- **Rhyme Density:** 70-80% of lines should rhyme naturally (no forced rhymes).
- **Syllable Count:** Maintain consistent meter (8-10 syllables per line in verses).
- **Wordplay:** Include 2-3 clever metaphors, puns, or literary devices.
- **Ad-libs & Slang:** Genre-appropriate ("Yeah", "Uh", "Ho!", "Check it").
- **Emotional Arc:** Build tension → climax → resolution.
- **Avoid Clichés:** NO overused phrases like "baby I love you" without context.
- **SUNO VOCALIZATION:** For non-English words, use simplified romanization with natural syllable breaks.

**4. PERFORMANCE CUES (Critical for Suno - Use ONLY these formats):**
IMPORTANT: Suno will NOT vocalize text inside asterisks or in ALL CAPS tags.
- Use *whispered* or *soft* for intimate moments
- Use *intense* or *powerful* for emotional peaks
- Use *melodic* or *autotuned* for modern pop/trap effects
- Use *breathy* for sensual verses
- Use *rap-style* or *spoken* for rap/spoken word
- Use *choir* or *layered* for anthemic chorus
- NEVER use parentheses like (whispered) - Suno will sing them!
- Place performance directions BEFORE the line, like:
  *whispered* Main tumse pyaar karta hoon
  *intense* Dil ki baat sun le

**5. STYLE STRING (For Suno Style Box):**
Format: [Genre], [BPM], [Key Instruments], [Mood/Vibe]
Example: "Trap/Hip-Hop, 140 BPM, 808s + Snare, Dark & Introspective"
Max 120 characters. Make it SPECIFIC and PRODUCTION-READY.

**6. GENIUS TOUCHES:**
- Internal rhymes within lines (not just end rhymes).
- Alliteration for memorability.
- Contrast verses (fast/slow, loud/quiet, high/low).
- Hook repetition with slight lyrical variations.
- Cultural/regional authenticity if applicable.
- For Hindi/Haryanvi: Use common, phonetically clear words (avoid overly complex Sanskrit terms).

**7. HINDI/HARYANVI SPECIAL RULES:**
- Prefer: "pyaar", "dil", "raat", "baat", "jaan", "khushi", "chaah"
- Avoid: Complex Sanskrit words, too many conjuncts (like "त्र", "ज्ञ")
- Break long words: "samjhauta" as "sam-jhau-ta" across multiple beats
- Use aspirated sounds naturally: "kh" (ख), "th" (थ), "ch" (छ), "ph" (फ)
- Emphasize vowel sounds: "aa" (आ), "ee" (ई), "oo" (ऊ) for better vocalization
- NEVER add English pronunciation guides in parentheses - they will be sung!

### OUTPUT STRUCTURE:
---
**STYLE STRING:** [Your optimized style description]
**BPM:** [Suggested tempo]
**KEY:** [Musical key recommendation]

[Intro]
[Lyrics here - use *performance-direction* before lines if needed]

[Verse 1]
[Lyrics here]

[Chorus]
[Lyrics here]

[Verse 2]
[Lyrics here]

[Bridge]
[Lyrics here]

[Outro]
[Lyrics here]

**PRODUCTION NOTES:**
- [Any special effects or layering notes]
- [General pronunciation guidance - NOT in lyrics]
---

Remember: Every lyric must be singable, emotionally resonant, and optimized for Suno's voice synthesis.
CRITICAL: Never use (parentheses) or [brackets] inside lyric lines - only use *asterisks* for cues.
"""

# --- 3.5. LYRIC QUALITY VALIDATOR ---
def validate_lyric_quality(lyrics_text):
    """Quick validation for lyric quality markers."""
    quality_checks = {
        "Has Structure Tags": any(tag in lyrics_text for tag in ["[Intro]", "[Verse", "[Chorus]", "[Bridge]"]),
        "Has Style String": "STYLE STRING:" in lyrics_text,
        "Has BPM Info": "BPM:" in lyrics_text,
        "No Devanagari": not any('\u0900' <= c <= '\u097F' for c in lyrics_text),  # Hindi script check
        "No Bengali Script": not any('\u0980' <= c <= '\u09FF' for c in lyrics_text),  # Bengali script check
        "Reasonable Length": 300 < len(lyrics_text) < 5000,
    }
    
    passed = sum(1 for v in quality_checks.values() if v)
    return quality_checks, passed, len(quality_checks)

# --- 3.6. EXTRACT & FORMAT RESPONSE ---
def extract_lyrics_from_response(response):
    """Safely extract text from API response object."""
    try:
        # Try to access .text attribute (most common)
        if hasattr(response, 'text') and response.text:
            return response.text.strip()
        # Try to access .content attribute
        elif hasattr(response, 'content') and response.content:
            return response.content.strip()
        # Try to convert to string
        else:
            text = str(response).strip()
            if len(text) > 50:  # Ensure it's not just object representation
                return text
            else:
                return None
    except Exception as e:
        print(f"Error extracting response: {e}")
        return None

# --- 3.7. CHECK IF RESPONSE IS COMPLETE ---
def is_response_complete(lyrics_text):
    """Check if the lyrics response appears to be complete."""
    # Must have multiple sections
    has_multiple_sections = lyrics_text.count('[') >= 5
    # Must have actual lyric content (not just headers)
    has_content = len(lyrics_text) > 800
    # Should end with [Outro] or PRODUCTION NOTES
    has_proper_ending = any(ending in lyrics_text for ending in ["[Outro]", "PRODUCTION NOTES:", "---"])
    
    return has_multiple_sections and has_content and has_proper_ending

# --- 3.8. POST-PROCESS LYRICS FOR SUNO VOCALIZATION ---
def optimize_for_suno_vocalization(lyrics_text, language):
    """Enhance lyrics for better Suno vocalization based on language."""
    # Remove any parenthetical performance cues that might have slipped through
    # Replace (whispered) with *whispered*, etc.
    performance_cues = [
        'whispered', 'screamed', 'autotune heavy', 'breathy', 'staccato',
        'choir backing', 'doubled vocals', 'intense', 'soft', 'melodic',
        'rap-style', 'spoken', 'powerful', 'layered', 'choir'
    ]
    
    for cue in performance_cues:
        # Replace (cue) with *cue* at the start of lines
        lyrics_text = re.sub(
            rf'\(({cue})\)',
            r'*\1*',
            lyrics_text,
            flags=re.IGNORECASE
        )
    
    # Remove any remaining parentheses with English text that might be vocalized
    # Keep only parentheses with actual lyric ad-libs like (yeah), (uh)
    lyrics_text = re.sub(
        r'\(note:.*?\)',
        '',
        lyrics_text,
        flags=re.IGNORECASE
    )
    
    # Remove inline phonetic guides like [py-aar], [dil] etc.
    lyrics_text = re.sub(r'\[[\w-]+\]', '', lyrics_text)
    
    if language in ["Hindi (Hinglish)", "Haryanvi (Desi)"]:
        # Add production note for Suno about Hindi pronunciation
        if "PRODUCTION NOTES:" in lyrics_text:
            production_note = """
- **Hindi/Haryanvi Vocalization Guide:**
  - Common words: pyaar (love), dil (heart), raat (night), baat (talk), jaan (life)
  - Emphasize vowel sounds: 'aa', 'ee', 'oo' for clarity
  - Keep consonants crisp: 'kh', 'th', 'ch', 'ph'
  - Multi-syllable words broken for rhythm: sam-jhau-ta, a-ke-la
  - No inline pronunciation guides used (to prevent vocalization)"""
            lyrics_text = lyrics_text.replace("PRODUCTION NOTES:", f"PRODUCTION NOTES:{production_note}")
    
    return lyrics_text

# --- 4. INPUT UI ---
st.title("🎹 Music Architect Pro")
st.markdown("Create world-class Suno AI song blueprints in seconds.")

st.subheader("🎯 Song Setup")
col_a, col_b = st.columns(2)

with col_a:
    topic = st.text_input("Song Topic / Concept", "Late Night City Drive")
    language = st.selectbox(
        "Language",
        [
            "English",
            "Hindi (Hinglish)",
            "Haryanvi (Desi)",
            "Punjabi (Romanized)",
            "Bengali (Romanized)",
            "Spanish",
            "French",
            "Other (Romanized)"
        ]
    )

with col_b:
    genre = st.selectbox(
        "Genre / Vibe",
        [
            "Pop",
            "Trap",
            "Lo-Fi",
            "R&B",
            "Rock",
            "EDM",
            "Cinematic",
            "Afrobeat",
            "Hip-Hop",
            "Indie"
        ]
    )
    voice = st.selectbox(
        "Vocal Style",
        [
            "Male",
            "Female",
            "Duo",
            "Choir",
            "Rap/Spoken",
            "Soft/Breathy",
            "Powerful"
        ]
    )

st.divider()

# --- 5. GENERATION LOGIC ---
if st.button("🚀 Architect Blueprint", type="primary"):
    try:
        # 1. Get the list of working models
        available_models = get_best_models()
        if not available_models:
            st.error("❌ Could not connect to Google AI. Check your API Key.")
            st.stop()
        
        status_box = st.empty()
        result_area = st.empty()
        
        prompt = f"""
        Create a GENIUS-LEVEL Suno AI song blueprint with world-class lyrics.
        
        TOPIC/CONCEPT: {topic}
        LANGUAGE: {language}
        GENRE/VIBE: {genre}
        VOCAL STYLE: {voice}
        
        REQUIREMENTS:
        1. Use the OUTPUT STRUCTURE with proper [Tags].
        2. Write lyrics that are CATCHY, MEMORABLE, and SINGABLE.
        3. Include 2-3 clever metaphors or wordplay elements.
        4. Maintain consistent rhyme schemes (AABB or ABAB).
        5. Add 3-5 performance cues using *asterisk* format (NOT parentheses).
        6. Create a unique STYLE STRING optimized for Suno v4.
        7. For non-English: Use ONLY Romanized Script (no Devanagari/Bengali/Arabic).
        8. Each verse should have 8-12 lines with natural flow.
        9. Chorus must be highly repetitive and catchy (4-6 lines).
        10. Include production notes for layering/effects.
        11. For Hindi/Haryanvi: Use phonetically clear words and break multi-syllable words into beats.
        12. CRITICAL: NEVER use (parentheses) or [brackets] inside lyric lines - only *asterisks* for cues.
        13. NO inline pronunciation guides like [py-aar] - put those in PRODUCTION NOTES only.
        
        Prioritize QUALITY over quantity. Every word must serve the song.
        """
        
        success = False
        max_retries = 2
        
        # 2. Try models one by one
        for model_name in available_models:
            if success:
                break  # Exit if already successful
                
            try:
                status_box.info(f"🤖 Contacting **{model_name}**...")
                
                # Retry logic for incomplete responses
                for attempt in range(max_retries + 1):
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_INSTRUCTION,
                                temperature=0.95,
                                max_output_tokens=4096,
                            )
                        )
                        
                        # Extract lyrics from response
                        lyrics_text = extract_lyrics_from_response(response)
                        
                        if not lyrics_text:
                            status_box.warning(f"⚠️ {model_name} returned empty response. Trying next model...")
                            break  # Try next model
                        
                        # Check if response is complete
                        if is_response_complete(lyrics_text):
                            # Optimize for Suno vocalization
                            lyrics_text = optimize_for_suno_vocalization(lyrics_text, language)
                            
                            status_box.success(f"✅ Generated using **{model_name}** (Attempt {attempt + 1})")
                            
                            # Validate quality
                            quality_check, passed, total = validate_lyric_quality(lyrics_text)
                            with st.expander("📊 Quality Validation"):
                                for check, result in quality_check.items():
                                    st.write(f"{'✅' if result else '⚠️'} {check}")
                                st.write(f"**Score: {passed}/{total}**")
                            
                            result_area.text_area("Your Genius Blueprint (Copy-Paste to Suno):", value=lyrics_text, height=600)
                            
                            # Download Button
                            st.download_button(
                                label="💾 Download Blueprint (.txt)",
                                data=lyrics_text,
                                file_name=f"Suno_Genius_{int(time.time())}.txt",
                                mime="text/plain"
                            )
                            success = True
                            break  # Exit retry loop
                        else:
                            # Response incomplete, retry
                            if attempt < max_retries:
                                status_box.warning(f"⚠️ Response incomplete. Retrying... (Attempt {attempt + 2}/{max_retries + 1})")
                                time.sleep(1)
                                continue
                            else:
                                status_box.warning(f"⚠️ {model_name} keeps returning incomplete responses. Trying next model...")
                                break  # Try next model
                    
                    except Exception as retry_error:
                        if attempt < max_retries:
                            status_box.warning(f"⚠️ Attempt {attempt + 1} failed. Retrying...")
                            time.sleep(1)
                            continue
                        else:
                            raise  # Re-raise to outer exception handler
                        
            except Exception as e:
                # Check for specific "Busy" errors
                error_str = str(e).lower()
                if "429" in error_str or "quota" in error_str:
                    status_box.warning(f"⚠️ {model_name} is out of quota. Switching...")
                elif "503" in error_str or "overloaded" in error_str:
                    status_box.warning(f"⚠️ {model_name} is overloaded. Switching...")
                else:
                    # Show full error for debugging
                    status_box.error(f"❌ Error with {model_name}: {str(e)[:200]}")
                continue  # Try next model
        
        # Final status check
        if not success:
            st.error("❌ Could not generate complete lyrics. All models either failed or returned incomplete responses.")
            with st.expander("🔧 Troubleshooting"):
                st.write("**Possible causes:**")
                st.write("1. Google API quota exceeded")
                st.write("2. Network connectivity issue")
                st.write("3. Invalid API key")
                st.write("4. Model output token limit too low")
                st.write("5. All models are currently unavailable")
                st.write("")
                st.write("**Try:**")
                st.write("- Wait 2-3 minutes and try again")
                st.write("- Check your Google API key is valid")
                st.write("- Ensure you have sufficient API quota")
                
    except Exception as e:
        st.error("❌ Unexpected error while generating. Please try again.")
        st.write(f"Error details: {str(e)[:200]}")
        with st.expander("🐛 Full Error Trace"):
            st.code(traceback.format_exc())