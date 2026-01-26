import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

# --- 1. CONFIG ---
st.set_page_config(page_title="Banking Exam Tutor", page_icon="🎓", layout="wide")

# Load Secrets
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
else:
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("GOOGLE_API_KEY")

# --- 2. SECURITY CHECK ---
if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("🔒 Please unlock the app on the 'Home' page first.")
    st.stop()

# --- 3. HELPER FUNCTIONS ---
def get_pdf_text(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache_data(show_spinner=False)
def generate_quiz(text, topic, difficulty):
    client = genai.Client(api_key=api_key)
    
    sys_instruct = (
        "You are a strict Exam Setter for Indian Banking Exams (SBI PO, IBPS PO). "
        "Your goal is to test the student's concept clarity and speed."
    )

    prompt = f"""
    Create a Short Quiz (5 Questions) based on the text below.
    
    Topic: {topic}
    Difficulty: {difficulty} (Make options tricky like real banking exams).
    
    SOURCE TEXT:
    {text[:5000]} (Truncated for speed)
    
    OUTPUT FORMAT:
    Question 1: [The Question]
    A) [Option]
    B) [Option]
    C) [Option]
    D) [Option]
    E) [Option]
    Answer: [Correct Option Letter] - [Short Explanation]
    
    (Repeat for 5 questions)
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(system_instruction=sys_instruct)
        )
        return response.text
    except Exception as e:
        return f"Error: {e}"

# --- 4. UI LAYOUT ---
st.title("🎓 SBI/IBPS Exam Auto-Quizzer")
st.markdown("Upload your notes (PDF), and AI will grill you with **Mains-Level Questions**.")

col1, col2 = st.columns([1, 2])

with col1:
    st.info("📂 **Upload Study Material**")
    uploaded_file = st.file_uploader("Upload PDF (Notes/Current Affairs)", type=["pdf"])
    
    difficulty = st.select_slider("🔥 Difficulty Level", options=["Clerk Prelims", "PO Prelims", "PO Mains (Hard)"])
    
    if uploaded_file:
        generate_btn = st.button("📝 Generate Quiz", type="primary", use_container_width=True)

with col2:
    if uploaded_file and generate_btn:
        with st.spinner("🧠 Analyzing PDF & Setting Paper..."):
            # 1. Read PDF
            raw_text = get_pdf_text(uploaded_file)
            
            # 2. Generate Quiz
            quiz_content = generate_quiz(raw_text, "Banking Aptitude", difficulty)
            
            # 3. Display Result
            st.success("✅ Paper Set! Solve these:")
            
            # Simple parsing to hide answers
            questions = quiz_content.split("Question")
            for q in questions[1:]: # Skip empty first split
                parts = q.split("Answer:")
                question_body = "Question" + parts[0]
                
                # Show Question
                st.markdown("---")
                st.markdown(f"### {question_body}")
                
                # Hidden Answer (Click to Reveal)
                with st.expander("👁️ View Answer & Explanation"):
                    if len(parts) > 1:
                        st.info(parts[1])
                    else:
                        st.warning("AI formatting error, check raw text below.")