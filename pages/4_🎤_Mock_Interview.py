import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- 1. CONFIG ---
st.set_page_config(page_title="SBI PO Mock Interview", page_icon="🎤", layout="centered")

# Load Secrets
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("GOOGLE_API_KEY")

# --- 2. SESSION STATE (Memory) ---
if "interview_active" not in st.session_state:
    st.session_state.interview_active = False
if "current_question" not in st.session_state:
    st.session_state.current_question = "Tell me about yourself and why you want to join the Banking sector?"

# --- 3. AI FUNCTIONS ---
@st.cache_data(show_spinner=False)
def get_feedback(audio_bytes, question):
    client = genai.Client(api_key=api_key)
    
    sys_instruct = (
        "You are a strict Senior Interviewer for the State Bank of India (SBI). "
        "Listen to the candidate's audio answer. "
        "Evaluate them on: 1. Content Accuracy 2. Communication Confidence 3. Relevance. "
        "Be professional but critical."
    )
    
    prompt = f"""
    QUESTION ASKED: "{question}"
    
    Please analyze the attached audio answer.
    Output format:
    **Score:** X/10
    **Strength:** [What they did well]
    **Weakness:** [Where they fumbled or lacked knowledge]
    **Better Answer:** [How a topper would answer this in 1 line]
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    parts=[
                        types.Part.from_text(prompt),
                        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
                    ]
                )
            ],
            config=types.GenerateContentConfig(system_instruction=sys_instruct)
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

def get_next_question():
    # Simple logic to rotate questions (You can make this AI-generated too!)
    questions = [
        "What is the difference between Repo Rate and Reverse Repo Rate?",
        "How would you handle an angry customer in a rural branch?",
        "What is your view on the privatization of Public Sector Banks?",
        "Explain the concept of NPA (Non-Performing Assets).",
        "Why did you choose Political Science if you wanted to work in a Bank?"
    ]
    import random
    return random.choice(questions)

# --- 4. UI LAYOUT ---
st.title("🎤 AI Mock Interview Panel")
st.markdown("Practice speaking clearly and confidently. The AI is listening.")

# Display Question
st.markdown("### 🗣️ Interviewer Asks:")
st.info(f"**{st.session_state.current_question}**")

# Audio Recorder (Native Streamlit)
audio_value = st.audio_input("Record your answer")

if audio_value:
    st.audio(audio_value)
    
    if st.button("Submit Answer for Evaluation", type="primary"):
        with st.spinner("🤔 The panel is discussing your answer..."):
            # Get raw bytes
            audio_bytes = audio_value.read()
            
            # Send to Gemini
            feedback = get_feedback(audio_bytes, st.session_state.current_question)
            
            st.success("Evaluation Complete")
            st.markdown("---")
            st.markdown(feedback)
            
            # Next Question Button
            if st.button("Next Question ➡️"):
                st.session_state.current_question = get_next_question()
                st.rerun()

# Sidebar Info
with st.sidebar:
    st.write("💡 **Tip:** Speak slowly and maintain a professional tone. Gemini can detect nervousness in your voice!")