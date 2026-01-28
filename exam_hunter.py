import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load Environment Variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    print("❌ Error: GOOGLE_API_KEY not found in .env file.")
    exit()

# 2. Initialize the Client
client = genai.Client(api_key=API_KEY)

class ExamHunterBot:
    def __init__(self):
        print("🚀 Exam Hunter AI Initialized")
        self.available_models = []
        self.refresh_models()

    def refresh_models(self):
        """
        Dynamically fetches the list of ALL models available to your API key.
        Sorts them so the 'Best' (Pro/Flash) are tried first.
        """
        print("🔄 Fetching available models from Google...")
        try:
            # Get all models that support generating content
            all_models = list(client.models.list())
            
            # Filter for models that generate text (Gemini models)
            valid_models = [m.name for m in all_models if "gemini" in m.name and "vision" not in m.name]
            
            # Sort them: Prioritize 'Pro' and 'Flash' versions
            # This lambda sorts models containing 'pro' or 'flash' to the top
            valid_models.sort(key=lambda x: 0 if "pro" in x else (1 if "flash" in x else 2))
            
            self.available_models = valid_models
            print(f"✅ Found {len(self.available_models)} active models: {', '.join(self.available_models[:3])}...")
        except Exception as e:
            print(f"⚠️ Could not auto-fetch models. Using fail-safe defaults. Error: {e}")
            self.available_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]

    def analyze_news_with_failover(self, prompt, use_search=False):
        """
        Tries every available model in the list until one works.
        """
        response = None
        tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else None

        # Loop through the dynamic list of models
        for model_name in self.available_models:
            # Clean model name (sometimes API returns 'models/gemini-pro', we need just 'gemini-pro')
            clean_model_name = model_name.replace("models/", "")
            
            try:
                print(f"🤖 Contacting {clean_model_name}...")
                
                response = client.models.generate_content(
                    model=clean_model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=tools,
                        response_mime_type="application/json" if not use_search else "text/plain"
                    )
                )
                print(f"✅ Success! Connected to {clean_model_name}.")
                return response
            
            except Exception as e:
                error_msg = str(e).lower()
                # Ignore common "busy" errors and keep trying
                if "429" in error_msg or "quota" in error_msg:
                    print(f"⚠️ {clean_model_name} quota exceeded. Switching...")
                elif "503" in error_msg or "overloaded" in error_msg:
                    print(f"⚠️ {clean_model_name} is busy. Switching...")
                elif "404" in error_msg:
                     print(f"⚠️ {clean_model_name} not found (skipping)...")
                else:
                    print(f"⚠️ Error with {clean_model_name}: {e}")
                continue

        print("❌ All available AI models failed. Please try again in 5 minutes.")
        return None

    def run_hunt(self, mode, query):
        print(f"\n🔍 Hunting for: {query} ({mode} mode)...")
        
        search_prompt = f"""
        ACT AS: Expert Bank Exam Tutor (IBPS/SBI PO).
        TASK: Search for CURRENT AFFAIRS and BANKING AWARENESS news for: {query}.
        
        STRICT RULES:
        1. Find 5-7 high-probability exam topics.
        2. Focus on: RBI Guidelines, Appointments, GDP Forecasts, Awards, Summits.
        3. IGNORE: Political debates, crime, movies.
        
        OUTPUT FORMAT (Strict JSON):
        [
            {{
                "topic": "Category (e.g., Banking/Awards)",
                "headline": "Short headline",
                "question": "A tricky exam question based on this?",
                "answer": "The specific answer",
                "explanation": "Why is this important?"
            }}
        ]
        """
        
        response = self.analyze_news_with_failover(search_prompt, use_search=True)
        
        if response and response.text:
            print("\n" + "="*40)
            print("🎓 EXAM HUNTER REPORT")
            print("="*40)
            print(response.text)
            print("="*40)
            
            with open("exam_report.txt", "w", encoding="utf-8") as f:
                f.write(response.text)
            print("💾 Report saved to 'exam_report.txt'")
        else:
            print("❌ No data received.")

# --- MAIN EXECUTION LOOP ---
if __name__ == "__main__":
    bot = ExamHunterBot()
    
    while True:
        print("\n" + "="*30)
        print("🎓 BANK EXAM GK HUNTER")
        print("1. 🔴 Scan Live News")
        print("2. 📅 Archive Search")
        print("3. ❌ Exit")
        
        choice = input("👉 Select Mode (1-3): ").strip()
        
        if choice == "1":
            bot.run_hunt("Live", "Important Banking & Economy news from the last 24 hours")
        elif choice == "2":
            month = input("👉 Enter Month & Year (e.g., 'December 2025'): ").strip()
            bot.run_hunt("Archive", f"Important Banking & Economy news for {month}")
        elif choice == "3":
            print("👋 Happy Studying!")
            break
        else:
            print("Invalid selection.")