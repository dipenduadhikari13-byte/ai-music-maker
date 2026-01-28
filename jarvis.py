import os
import sys
import json
import datetime
import pyttsx3
import requests
import threading
from dotenv import load_dotenv
from googlesearch import search

# --- IMPORT YOUR MODULES ---
from hearing import SmartEar
from vision_direct import VisionAgentDirect

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("❌ CRITICAL ERROR: GOOGLE_API_KEY not found in .env")
    sys.exit(1)

class Jarvis:
    def __init__(self):
        print("⚙️ Initializing Jarvis System (Direct Mode)...")
        
        # 1. The Voice
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 170)
            self.engine.setProperty('volume', 1.0)
        except Exception as e:
            print(f"⚠️ Voice Engine Warning: {e}")

        # 2. The Body Parts
        self.ear = SmartEar()
        self.ear.calibrate() 
        self.eyes = VisionAgentDirect()

        # 3. The Brain (Model Priority List)
        self.available_models = [
            "gemini-2.0-flash-lite-preview-02-05", 
            "gemini-2.5-flash", 
            "gemini-1.5-flash"
        ]
        
        print("✅ Jarvis is Online. Waiting for commands.")
        self.speak("Systems online. Ready.")

    def speak(self, text):
        """Talks back to the user."""
        print(f"🤖 Jarvis: {text}")
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"❌ Audio Error: {e}")

    def ask_gemini_direct(self, prompt, structure_json=False):
        """Sends raw HTTP request to Google, bypassing the buggy library."""
        url_base = "https://generativelanguage.googleapis.com/v1beta/models/"
        
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        # If we need JSON output (for the Router), we enforce it here
        if structure_json:
            payload["generationConfig"] = {"response_mime_type": "application/json"}

        # Loop through models until one works
        for model in self.available_models:
            try:
                full_url = f"{url_base}{model}:generateContent?key={API_KEY}"
                response = requests.post(full_url, headers=headers, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    return result['candidates'][0]['content']['parts'][0]['text']
                elif response.status_code == 429:
                    continue # Rate limit, try next model
                elif response.status_code == 404:
                    continue # Model not found, try next
                else:
                    print(f"   (Error {response.status_code} on {model})")
            except:
                continue
                
        return None

    def search_web(self, query):
        print(f"🌍 Searching Google for: {query}...")
        try:
            results = list(search(query, num_results=3, advanced=True))
            if not results: return "No results found."
            context = " ".join([f"{r.title}: {r.description}" for r in results])
            return context[:2000]
        except Exception as e:
            print(f"❌ Search Error: {e}")
            return "Internet search failed."

    def decide_action(self, user_text):
        today = datetime.date.today().strftime("%B %d, %Y")
        
        prompt = f"""
        Current Date: {today}.
        User said: "{user_text}"
        
        Classify intent:
        1. "search" -> News, facts, 'who is', 'what is', weather.
        2. "click" -> UI actions (click, open, type).
        3. "chat" -> Casual conversation.
        
        Return ONLY valid JSON:
        {{ "type": "search", "query": "..." }}
        {{ "type": "click", "target": "..." }}
        {{ "type": "chat", "response": "..." }}
        """
        
        raw_response = self.ask_gemini_direct(prompt, structure_json=True)
        
        if raw_response:
            try:
                # Clean Markdown if present
                clean = raw_response.replace("```json", "").replace("```", "").strip()
                return json.loads(clean)
            except:
                pass
        
        return {"type": "chat", "response": "I couldn't process that command."}

    def generate_final_answer(self, user_question, google_context):
        prompt = f"""
        User Question: "{user_question}"
        Search Results: "{google_context}"
        
        Task: Answer the question using the Search Results. 
        Style: Concise, like Iron Man's Jarvis. 1-2 sentences max.
        """
        answer = self.ask_gemini_direct(prompt)
        return answer if answer else "I found info but couldn't read it."

    def run(self):
        while True:
            user_text = self.ear.listen()
            if not user_text: continue
            
            if "exit" in user_text.lower():
                self.speak("Goodbye.")
                break

            decision = self.decide_action(user_text)
            intent = decision.get("type")
            print(f"🧠 Intent: {intent}")
            
            if intent == "search":
                query = decision.get("query")
                self.speak(f"Searching for {query}...")
                context = self.search_web(query)
                answer = self