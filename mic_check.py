import sounddevice as sd
import numpy as np

def print_sound_level(indata, frames, time, status):
    """Prints a live volume bar to the terminal."""
    volume_norm = np.linalg.norm(indata) * 10
    
    # Create a visual bar
    bar_length = int(volume_norm)
    bar = "█" * bar_length
    
    # Print overwrite (keeps it on one line)
    print(f"|{bar:<50}| Level: {volume_norm:.2f}", end="\r")

def list_devices():
    print("\n🔍 Scanning Audio Devices...")
    print(sd.query_devices())
    
    default_input = sd.query_devices(kind='input')
    print(f"\n✅ Default Input Device: '{default_input['name']}' (Index: {default_input['index']})")

if __name__ == "__main__":
    list_devices()
    
    print("\n🎤 STARTING LIVE TEST (Press Ctrl+C to stop)")
    print("Speak into your mic to see the bars move!")
    print("-" * 60)
    
    try:
        # 1. Listen to the default device
        with sd.InputStream(callback=print_sound_level):
            while True:
                sd.sleep(100)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Try changing the device index in the code.")