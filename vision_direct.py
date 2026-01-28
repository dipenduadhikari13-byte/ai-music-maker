import os
import time
import json
import base64
import requests
import pyautogui
from io import BytesIO
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("❌ ERROR: GOOGLE_API_KEY not found in .env file!")
class VisionAgentDirect:
    def __init__(self):
        # Fail-safe: Slam mouse to corner to stop script
        pyautogui.FAILSAFE = True
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"✅ Vision Agent Initialized. Screen: {self.screen_width}x{self.screen_height}")

    def capture_screen_base64(self):
        """Captures screen and converts it to base64 for the API."""
        screenshot = pyautogui.screenshot()
        buffered = BytesIO()
        screenshot.save(buffered, format="JPEG", quality=70) # Lower quality = Faster speed
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_coordinates_direct(self, user_instruction):
        print(f"🧠 Thinking about: '{user_instruction}'...")
        
        b64_image = self.capture_screen_base64()
        
        # --- MODEL LIST FROM YOUR ACCOUNT ---
        # We try these in order. 'Lite' first because it rarely hits rate limits.
        models_to_try = [
            "gemini-2.0-flash-lite-preview-02-05", # Fastest & Likely Free
            "gemini-2.5-flash",                     # Newest High-Perf
            "gemini-flash-latest",                  # Standard Fallback
            "gemini-2.0-flash"                      # Backup
        ]

        headers = {"Content-Type": "application/json"}
        
        prompt_text = f"""
        You are a computer automation agent.
        User wants: "{user_instruction}"
        Look at the screenshot. Find the UI element (icon/button/bar) to click.
        Return ONLY a JSON object with coordinates (0-1000 scale).
        Format: {{ "x": 500, "y": 500, "description": "element name" }}
        """

        data = {
            "contents": [{
                "parts": [
                    {"text": prompt_text},
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64_image}}
                ]
            }],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }

        # --- RETRY LOOP ---
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
            
            try:
                # print(f"   Trying model: {model}...") 
                response = requests.post(url, headers=headers, json=data)
                
                if response.status_code == 200:
                    # Success! Parse the JSON.
                    result_json = response.json()
                    text_content = result_json['candidates'][0]['content']['parts'][0]['text']
                    return json.loads(text_content)
                
                elif response.status_code == 429:
                    print(f"   ⚠️ Too busy ({model}). Switching to next model...")
                    continue # Try next model
                elif response.status_code == 404:
                    print(f"   ⚠️ Not found ({model}). Switching...")
                    continue
                else:
                    print(f"   ❌ Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"   ❌ Connection Error on {model}: {e}")

        print("❌ All models failed.")
        return None

    def execute_click(self, relative_x, relative_y):
        target_x = int((relative_x / 1000) * self.screen_width)
        target_y = int((relative_y / 1000) * self.screen_height)
        
        print(f"🖱️ Clicking at: {target_x}, {target_y}")
        pyautogui.moveTo(target_x, target_y, duration=0.5)
        pyautogui.click()

    def run(self, command):
        result = self.get_coordinates_direct(command)
        
        if result:
            print(f"🎯 Target identified: {result.get('description')}")
            self.execute_click(result['x'], result['y'])
        else:
            print("❌ Could not find target.")

if __name__ == "__main__":
    agent = VisionAgentDirect()
    
    print("⏳ Switching to browser in 3 seconds...")
    time.sleep(3)
    
    agent.run("Click the search bar")