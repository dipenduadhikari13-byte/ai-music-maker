import streamlit as st
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Exam Hunter",
    page_icon="🎓",
    layout="wide"
)

# --- 1. SECURE API KEY ---
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

# --- 2. THE EXAM TUTOR BRAIN ---
SYSTEM_INSTRUCTION = """
You are "Exam Hunter," an expert tutor for Indian Banking Exams (IBPS PO, SBI PO, RBI Grade B).
Your goal is to provide concise, exam-relevant content.

MODES:
1. **Simplify:** Take a complex topic (like RBI Circulars) and break it down into bullet points.
2. **Quiz:** Generate 5 Multiple Choice Questions (MCQs) with answers and explanations on a topic.
3. **Strategy:** Provide a study plan or tips for specific sections (Quant/Reasoning/English).

STYLE:
- Professional, direct, and encouraging.
- Use bolding for key terms.
- Always assume the user is short on time.
"""

# --- 3. UI INTERFACE ---
st.title("🎓 Exam Hunter: Banking Edition")
st.markdown("### *Your AI Tutor for IBPS & SBI PO*")

# Sidebar for Mode Selection
mode = st.sidebar.radio(
    "Choose Tool:",
    ["📰 Topic Simplifier", "❓ MCQ Generator", "📅 Study Strategist"]
)

# --- MAIN CONTENT AREA ---

if mode == "📰 Topic Simplifier":
    st.subheader("Simplify Complex Topics")
    topic_input = st.text_area("Paste text (e.g., News Article, RBI Circular) or type a Topic:", height=150, placeholder="Example: What is the new UPI Lite limit?")
    
    if st.button("🔍 Simplify"):
        prompt = f"Simplify this for a Banking Aspirant. Highlight key data (amounts, dates, committees):\n\n{topic_input}"
        with st.spinner("Analyzing..."):
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
            )
            st.markdown(response.text)

elif mode == "❓ MCQ Generator":
    st.subheader("Generate Practice Questions")
    col1, col2 = st.columns(2)
    with col1:
        subject = st.selectbox("Subject", ["General Awareness", "Quantitative Aptitude", "Reasoning Ability", "English"])
    with col2:
        difficulty = st.selectbox("Difficulty", ["Clerk Level (Easy)", "PO Prelims (Medium)", "PO Mains (Hard)"])
    
    specific_topic = st.text_input("Specific Topic (Optional)", "e.g., Current Affairs Jan 2026, Syllogism, DI")
    
    if st.button("🎲 Generate MCQs"):
        prompt = f"Create 5 {difficulty} level MCQs for {subject}. Topic: {specific_topic}. Format: Question, Options, Correct Answer, Explanation."
        with st.spinner("Setting Paper..."):
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
            )
            st.markdown(response.text)

elif mode == "📅 Study Strategist":
    st.subheader("Get a Custom Plan")
    hours = st.slider("Hours available today:", 1, 12, 4)
    weakness = st.text_input("My Weakness is:", "Speed in Puzzles")
    
    if st.button("📝 Make Plan"):
        prompt = f"Create a {hours}-hour study schedule for today. My weakness is {weakness}. Focus on ROI (Return on Investment) topics."
        with st.spinner("Planning..."):
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)
            )
            st.markdown(response.text)