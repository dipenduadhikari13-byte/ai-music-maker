import os
import time
import queue
import collections
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from groq import Groq
from dotenv import load_dotenv

# --- CONFIGURATION ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ ERROR: GROQ_API_KEY not found in .env file!")

# Settings
SAMPLE_RATE = 44100
CHANNELS = 1
PRE_BUFFER_SECONDS = 0.5  # Keep 0.5s of audio BEFORE trigger
POST_BUFFER_SECONDS = 1.5 # Wait 1.5s of silence before stopping

class SmartEar:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
        self.q = queue.Queue()
        self.threshold = 20
        
        # Ring Buffer: Stores the last N chunks (0.5 seconds of audio)
        # Calculates how many chunks fit in PRE_BUFFER_SECONDS
        # Assuming blocksize is roughly 0.1s (adjustable by OS)
        self.pre_buffer = collections.deque(maxlen=20) 
        
        print("✅ Smart Ear Initialized (Pre-Buffer Active).")

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.q.put(indata.copy())

    def calibrate(self):
        print("\n🎤 Calibrating... (STAY SILENT)")
        with self.q.mutex:
            self.q.queue.clear()
            
        noise_levels = []
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.callback):
            start = time.time()
            while time.time() - start < 1.0:
                data = self.q.get()
                volume = np.linalg.norm(data) * 10
                noise_levels.append(volume)
        
        avg_noise = sum(noise_levels) / len(noise_levels) if noise_levels else 0
        self.threshold = max(15, avg_noise * 1.5) # Minimum 15 to prevent sensitivity madness
        print(f"✅ Threshold set to: {self.threshold:.2f}")

    def listen(self):
        print(f"\n👂 Listening... (Threshold: {self.threshold:.1f})")
        
        recording_data = []
        is_speaking = False
        start_time = time.time()
        last_sound_time = time.time()

        with self.q.mutex:
            self.q.queue.clear()
            self.pre_buffer.clear()

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.callback):
            while True:
                data = self.q.get()
                volume = np.linalg.norm(data) * 10
                
                if volume > self.threshold:
                    if not is_speaking:
                        print("🔴 Recording...")
                        is_speaking = True
                        # 1. ADD THE PAST (Pre-Buffer) to the start
                        recording_data.extend(self.pre_buffer)
                    
                    last_sound_time = time.time()
                    recording_data.append(data)
                
                elif is_speaking:
                    # 2. ADD THE SILENCE (Context)
                    recording_data.append(data)
                    
                    if (time.time() - last_sound_time) > POST_BUFFER_SECONDS:
                        print("✅ Finished.")
                        break
                else:
                    # 3. WE ARE WAITING: Save to Pre-Buffer
                    self.pre_buffer.append(data)
                
                if is_speaking and (time.time() - start_time > 15):
                    break

        if not recording_data:
            return None

        # Transcribe
        full_audio = np.concatenate(recording_data, axis=0)
        wav.write("temp_voice.wav", SAMPLE_RATE, full_audio)

        print("⏳ Transcribing...")
        try:
            with open("temp_voice.wav", "rb") as file_obj:
                transcription = self.client.audio.transcriptions.create(
                    file=("temp_voice.wav", file_obj.read()),
                    model="whisper-large-v3",
                    prompt="The user is giving commands to an AI assistant.", # Context Hint
                    language="en",
                    temperature=0.0
                )
            text = transcription.text.strip()
            print(f"🗣️ You said: '{text}'")
            return text
        except Exception as e:
            print(f"❌ Error: {e}")
            return None

if __name__ == "__main__":
    ear = SmartEar()
    ear.calibrate()
    
    while True:
        try:
            ear.listen()
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break