import sounddevice as sd
import numpy as np
import time

def test_audio_flow():
    print("Testing audio flow...")
    
    def print_audio_level(indata, outdata, frames, time, status):
        volume = np.sqrt(np.mean(indata**2))
        if volume > 0.01:
            print(f"🔊 Audio detected: {volume:.3f}")
        else:
            print(f"🔇 Silent: {volume:.3f}")
    
    print("Play some music and watch for audio levels...")
    with sd.Stream(callback=print_audio_level, channels=1):
        input("Press Enter to stop...")

if __name__ == "__main__":
    test_audio_flow()