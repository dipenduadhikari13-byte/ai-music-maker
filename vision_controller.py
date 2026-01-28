import time
import json
import pyautogui
from PIL import Image
from google import genai
from google.genai import types

# --- CONFIGURATION ---
# PASTE YOUR KEY HERE
API_KEY = "AIzaSyCGtclbvMgrrQvsAI6tjALTXNMrJToDYGo" 

class VisionAgent:
    def __init__(self):
        pyautogui.FAILSAFE = True 
        
        if "PASTE_YOUR" in API_KEY:
            raise ValueError("❌ STOP! You forgot to paste your API Key in line 10.")
            
        self.client = genai.Client(api_key=API_KEY)
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"✅ Vision Agent Initialized. Screen: {self.screen_width}x{self.screen_height}")

    def capture_screen(self, path="screenshot.png"):
        screenshot = pyautogui.screenshot()
        screenshot.save(path)
        return path

    def get_coordinates_from_ai(self, user_instruction, image_path):
        print(f"🧠 Thinking about: '{user_instruction}'...")
        image = Image.open(image_path)
        
        prompt = f"""
        You are a UI automation agent. Look at this screenshot.
        The user wants to: "{user_instruction}"
        Find the exact UI element to click.
        Return a JSON object with the coordinates (0-1000 scale).
        Format: {{ "x": 500, "y": 500, "description": "element name" }}
        """

        # --- THE FIX: MODEL HOPPING ---
        # We try these models in order. If one fails (404/429), we try the next.
        model_list = [
            "gemini-1.5-flash-latest", # Most likely to work
            "gemini-1.5-flash",        # Standard alias
            "gemini-1.5-flash-001",    # Specific version
            "gemini-2.0-flash-exp",    # Experimental (Fastest but quota limited)
        ]

        for model_name in model_list:
            try:
                # print(f"   Trying model: {model_name}...") 
                response = self.client.models.generate_content(
                    model=model_name, 
                    contents=[prompt, image],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                return json.loads(response.text)

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    print(f"   ⚠️ Rate limit on {model_name}. Switching...")
                elif "404" in error_msg:
                    print(f"   ⚠️ {model_name} not found. Switching...")
                else:
                    print(f"   ❌ Error on {model_name}: {e}")
        
        print("❌ All models failed. Check your API Key or Internet.")
        return None

    def execute_click(self, relative_x, relative_y):
        target_x = int((relative_x / 1000) * self.screen_width)
        target_y = int((relative_y / 1000) * self.screen_height)
        
        print(f"🖱️ Clicking at: {target_x}, {target_y}")
        pyautogui.moveTo(target_x, target_y, duration=1.0)
        pyautogui.click()

    def run(self, command):
        img_path = self.capture_screen()
        result = self.get_coordinates_from_ai(command, img_path)
        
        if result:
            print(f"🎯 Target identified: {result.get('description')}")
            self.execute_click(result['x'], result['y'])
        else:
            print("❌ Could not find target.")

if __name__ == "__main__":
    agent = VisionAgent()
    print("⏳ Switching to browser in 3 seconds...")
    time.sleep(3)
    agent.run("Click the search bar")