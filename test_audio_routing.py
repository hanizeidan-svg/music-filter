import sounddevice as sd
import numpy as np

def list_audio_devices():
    print("=== Audio Devices ===")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(f"{i}: {dev['name']}")
        print(f"   Inputs: {dev['max_input_channels']}, Outputs: {dev['max_output_channels']}")
        if 'stereo mix' in dev['name'].lower() or 'what u hear' in dev['name'].lower():
            print("   *** This might work for system audio capture ***")

def test_device(device_id):
    print(f"\n=== Testing Device {device_id} ===")
    try:
        duration = 3  # seconds
        print(f"Recording for {duration} seconds...")
        
        recording = sd.rec(int(duration * 44100), samplerate=44100, 
                          channels=1, device=device_id, dtype='float32')
        sd.wait()
        
        max_volume = np.max(np.abs(recording))
        print(f"Max volume detected: {max_volume}")
        
        if max_volume > 0.01:
            print("✅ Device is capturing audio!")
        else:
            print("❌ No audio detected - try playing music while testing")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    list_audio_devices()
    device_id = int(input("\nEnter device ID to test: "))
    test_device(device_id)