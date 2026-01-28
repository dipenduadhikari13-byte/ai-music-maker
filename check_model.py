import requests
import os

# --- PASTE YOUR KEY HERE ---
API_KEY = "AIzaSyCGtclbvMgrrQvsAI6tjALTXNMrJToDYGo"

def list_models():
    print(f"🔍 Checking available models for Key: {API_KEY[:10]}...")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "error" in data:
            print("\n❌ API Error:")
            print(data["error"]["message"])
            return

        print("\n✅ AVAILABLE MODELS:")
        found_any = False
        if "models" in data:
            for model in data["models"]:
                # We only care about models that can 'generateContent'
                if "generateContent" in model.get("supportedGenerationMethods", []):
                    print(f" - {model['name'].replace('models/', '')}")
                    found_any = True
        
        if not found_any:
            print("⚠️ No models found! Your project might be restricted.")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    list_models()