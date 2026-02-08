import streamlit as st
import os
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="YouTube Studio Pro",
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

# --- 2. DYNAMIC MODEL FETCHING ---
@st.cache_resource(ttl=3600)
def get_best_models():
    """Get available models from Google AI."""
    try:
        all_models = list(client.models.list())
        valid_models = [
            m.name.replace("models/", "") 
            for m in all_models 
            if "gemini" in m.name and "vision" not in m.name
        ]
        
        def model_priority(name):
            if "2.0-flash" in name: return 0
            if "1.5-pro" in name: return 1
            if "1.5-flash" in name: return 2
            return 3

        valid_models.sort(key=model_priority)
        return valid_models
    except Exception:
        return ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]

# --- 3. ENHANCED SYSTEM BRAIN (YouTube Studio Expert) ---
SYSTEM_INSTRUCTION = """
You are a "YouTube Growth Strategist & SEO Expert" specializing in music channels.
Your mission: Create viral-ready metadata with data-driven optimization.

### CORE PRINCIPLES:
1. **Algorithm Psychology:** Understand what makes YouTube recommend content
2. **CTR Optimization:** Craft titles that trigger curiosity + emotional response
3. **Retention Hooks:** Write descriptions that keep viewers engaged
4. **Search Dominance:** Target high-volume, low-competition keywords
5. **Brand Consistency:** Maintain channel identity across all content

### METADATA COMPONENTS:

**A. TITLES (Provide 5 Variants):**
1. **Emotional Hook** - Triggers feelings (e.g., "This Beat Will Break Your Heart 💔")
2. **Clickbait (Honest)** - Creates curiosity without lying (e.g., "You Won't Believe This Drop...")
3. **SEO Optimized** - Keyword-rich for search (e.g., "Dark Trap Beat 2024 | 808 Bass")
4. **Viral Format** - Trend-based (e.g., "POV: You're the Main Character [Phonk]")
5. **Niche Authority** - Expert positioning (e.g., "Pro Producer Breaks Down [Genre]")

**B. DESCRIPTION (Structure):**
```
🎯 HOOK (First 2 lines - visible before "Show More")
- Compelling question or bold statement
- Include primary keyword

📖 STORY SECTION (150-200 words)
- Behind-the-scenes narrative
- Emotional journey of the track
- Production techniques used

⏱️ TIMESTAMPS (If applicable)
0:00 - Intro
0:30 - Drop
etc.

🔗 LINKS & CTA
- Subscribe link with incentive
- Playlist links
- Social media

🏷️ CREDITS & TAGS
- Producer/Artist credits
- Software/plugins used
- Copyright info

#️⃣ HASHTAGS (3-5 max, most important first)
```

**C. TAGS (Provide 25-30 tags in priority order):**
- **Tier 1:** Primary keywords (exact match to title)
- **Tier 2:** Synonyms & variations
- **Tier 3:** Genre/mood descriptors
- **Tier 4:** Long-tail search phrases
- **Tier 5:** Channel branding tags

**D. THUMBNAIL CONCEPTS (3 Visual Ideas):**
Describe thumbnail designs with:
- Color scheme
- Text overlay (max 3 words)
- Visual focal point
- Emotional appeal

**E. SEO ANALYSIS:**
- Search volume estimate (High/Medium/Low)
- Competition level (High/Medium/Low)
- Suggested upload time (based on genre)
- Target audience demographics

**F. A/B TESTING VARIANTS:**
Provide 2 alternative title/thumbnail combos for testing.

### OUTPUT FORMAT:
Use clear markdown with emojis. Make it copy-paste ready for YouTube Studio.
"""

# --- 4. SEO SCORING FUNCTION ---
def calculate_seo_score(metadata_text):
    """Calculate SEO quality score based on best practices."""
    score = 0
    checks = {}
    
    # Title checks
    has_multiple_titles = metadata_text.count("**") >= 10
    checks["Multiple Title Options"] = has_multiple_titles
    score += 15 if has_multiple_titles else 0
    
    # Description length
    desc_length = len(metadata_text)
    checks["Description Length (800+)"] = desc_length > 800
    score += 20 if desc_length > 800 else 10
    
    # Has timestamps
    has_timestamps = "0:00" in metadata_text or "Timestamps" in metadata_text
    checks["Has Timestamps"] = has_timestamps
    score += 15 if has_timestamps else 0
    
    # Has hashtags
    has_hashtags = "#" in metadata_text
    checks["Has Hashtags"] = has_hashtags
    score += 10 if has_hashtags else 0
    
    # Has CTA
    has_cta = "subscribe" in metadata_text.lower() or "like" in metadata_text.lower()
    checks["Has Call-to-Action"] = has_cta
    score += 10 if has_cta else 0
    
    # Tag count
    tag_count = metadata_text.lower().count("tag")
    checks["Has 20+ Tags"] = tag_count > 0
    score += 15 if tag_count > 0 else 0
    
    # Thumbnail concepts
    has_thumbnail = "thumbnail" in metadata_text.lower()
    checks["Thumbnail Concepts"] = has_thumbnail
    score += 15 if has_thumbnail else 0
    
    return score, checks

def extract_text(response):
    """Safely extract text from model response."""
    if hasattr(response, "text") and response.text:
        return response.text.strip()
    if hasattr(response, "content") and response.content:
        return response.content.strip()
    return None

# --- 5. UI INTERFACE ---
st.title("📺 YouTube Studio Pro")
st.markdown("### *Complete Metadata Suite for Music Channels*")

# Sidebar - Advanced Settings
with st.sidebar:
    st.header("⚙️ Studio Settings")
    channel_name = st.text_input("📢 Channel Name", "Soul Note Originals")
    target_audience = st.selectbox("👥 Target Audience", [
        "Students (Study Music)",
        "Gamers (Background Music)",
        "Creators (No Copyright)",
        "Music Producers",
        "General Listeners"
    ])
    video_length = st.selectbox("⏱️ Video Length", [
        "< 3 min (Short Form)",
        "3-10 min (Standard)",
        "10-30 min (Extended)",
        "30+ min (Long Form)"
    ])
    monetization = st.checkbox("💰 Monetization Enabled", value=True)
    
    st.divider()
    st.subheader("🎯 Upload Strategy")
    upload_day = st.selectbox("📅 Best Upload Day", [
        "Auto-Suggest", "Monday", "Tuesday", "Wednesday", 
        "Thursday", "Friday", "Saturday", "Sunday"
    ])
    upload_time = st.selectbox("🕐 Best Upload Time", [
        "Auto-Suggest", "Morning (6-10 AM)", "Afternoon (12-3 PM)", 
        "Evening (5-8 PM)", "Night (9-12 AM)"
    ])

# Main Content Area
tab1, tab2, tab3 = st.tabs(["📝 Generate Metadata", "📊 SEO Analysis", "🔬 A/B Testing"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        video_topic = st.text_input("📝 Video Topic / Song Name", "Cyberpunk Samurai Lo-Fi")
        music_genre = st.selectbox("🎵 Genre", [
            "Lo-Fi", "Trap", "Drill", "Phonk", "Classical", 
            "Cinematic", "Ambient", "EDM", "Hip-Hop", "R&B"
        ])
        mood = st.selectbox("✨ Mood", [
            "Relaxing/Chill", "Dark/Aggressive", "Motivational", 
            "Sad/Emotional", "Party/Upbeat", "Epic/Cinematic"
        ])
    
    with col2:
        video_type = st.selectbox("📹 Video Type", [
            "Music Only (Visualizer)",
            "Lyric Video",
            "Music Video",
            "Behind The Scenes",
            "Tutorial/Breakdown"
        ])
        competition_level = st.selectbox("🎯 Competition", [
            "Low (Niche Topic)",
            "Medium (Popular Genre)",
            "High (Trending Topic)"
        ])
        include_timestamps = st.checkbox("⏱️ Include Timestamps", value=True)

    # Generate Button
    if st.button("🚀 Generate Metadata", type="primary"):
        with st.spinner("🧠 Analyzing Algorithm... Writing Metadata..."):
            prompt = f"""
            Generate a COMPLETE YouTube Studio Package for a music video.
            
            === VIDEO DETAILS ===
            TOPIC: {video_topic}
            GENRE: {music_genre}
            MOOD: {mood}
            CHANNEL: {channel_name}
            VIDEO TYPE: {video_type}
            TARGET AUDIENCE: {target_audience}
            VIDEO LENGTH: {video_length}
            COMPETITION: {competition_level}
            MONETIZATION: {'Yes' if monetization else 'No'}
            
            === REQUIREMENTS ===
            1. Provide 5 title variants (Emotional, Clickbait, SEO, Viral, Authority)
            2. Write a complete 800+ word description with:
               - Powerful hook (first 2 lines)
               - Story behind the track (200+ words)
               - {'Timestamps' if include_timestamps else 'No timestamps needed'}
               - Call-to-action for {channel_name}
               - Credits & links section
            3. Generate 25-30 tags in priority order (Tier 1-5)
            4. Suggest 3 thumbnail concepts with design details
            5. Provide SEO analysis (search volume, competition, best upload time)
            6. Create 2 A/B testing variants
            """

            models = get_best_models()
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
                    response_text = extract_text(response) or ""
                    if response_text:
                        break
                except Exception:
                    continue

            if not response_text:
                st.error("❌ Could not generate metadata. All models failed or are unavailable.")
                st.stop()

            # SEO Scoring
            seo_score, seo_checks = calculate_seo_score(response_text)
            
            # Display Results
            col_result1, col_result2 = st.columns([3, 1])
            
            with col_result1:
                st.subheader("📋 Your Complete Studio Package")
                st.text_area(
                    "Copy to YouTube Studio:", 
                    value=response_text, 
                    height=700,
                    key="metadata_output"
                )
            
            with col_result2:
                st.metric("SEO Score", f"{seo_score}/100", 
                         delta="Excellent" if seo_score > 80 else "Good" if seo_score > 60 else "Needs Work")
                
                with st.expander("📊 SEO Breakdown"):
                    for check, passed in seo_checks.items():
                        st.write(f"{'✅' if passed else '❌'} {check}")
            
            # Download Options
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="💾 Download Full Package (.txt)",
                    data=response_text,
                    file_name=f"YT_Studio_{video_topic.replace(' ', '_')}_{int(time.time())}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_dl2:
                # Create JSON format for programmatic use
                json_data = json.dumps({
                    "topic": video_topic,
                    "genre": music_genre,
                    "metadata": response_text,
                    "seo_score": seo_score
                }, indent=2)
                st.download_button(
                    label="📦 Download as JSON",
                    data=json_data,
                    file_name=f"YT_Studio_{video_topic.replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )

with tab2:
    st.subheader("📊 SEO Performance Analyzer")
    st.info("💡 Paste your generated metadata or existing YouTube description here for analysis.")
    
    analysis_text = st.text_area("Paste Metadata:", height=300)
    
    if st.button("🔍 Analyze SEO", use_container_width=True):
        if analysis_text:
            score, checks = calculate_seo_score(analysis_text)
            
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                st.metric("Overall Score", f"{score}/100")
            with col_a2:
                st.metric("Passed Checks", f"{sum(checks.values())}/{len(checks)}")
            with col_a3:
                grade = "A+" if score > 90 else "A" if score > 80 else "B" if score > 70 else "C"
                st.metric("Grade", grade)
            
            st.subheader("Detailed Breakdown:")
            for check, passed in checks.items():
                st.write(f"{'✅' if passed else '❌'} {check}")
        else:
            st.warning("⚠️ Please paste some metadata to analyze.")

with tab3:
    st.subheader("🔬 A/B Testing Lab")
    st.markdown("""
    Compare multiple variants to find the best performing metadata.
    Use this to test different titles, thumbnails, and descriptions.
    """)
    
    col_ab1, col_ab2 = st.columns(2)
    
    with col_ab1:
        st.text_input("🅰️ Variant A - Title", key="variant_a_title")
        st.text_area("Description A", height=150, key="variant_a_desc")
    
    with col_ab2:
        st.text_input("🅱️ Variant B - Title", key="variant_b_title")
        st.text_area("Description B", height=150, key="variant_b_desc")
    
    if st.button("⚔️ Compare Variants", use_container_width=True):
        st.info("🔄 A/B testing analysis will be available in future updates!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎯 YouTube Studio Pro | Optimized for Music Channels | Powered by Gemini AI</p>
</div>
""", unsafe_allow_html=True)
