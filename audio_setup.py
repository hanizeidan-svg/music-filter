import sounddevice as sd
import numpy as np

def list_devices():
    print("=== AUDIO DEVICES ===")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        print(f"{i}: {dev['name']}")
        print(f"   Inputs: {dev['max_input_channels']}, Outputs: {dev['max_output_channels']}")
        if dev['max_input_channels'] > 0 and dev['max_output_channels'] > 0:
            print("   *** Can be both input and output ***")

def test_device(device_id, is_input=True):
    try:
        if is_input:
            print(f"Testing INPUT device {device_id}...")
            duration = 3
            print("Recording for 3 seconds...")
            recording = sd.rec(int(duration * 44100), samplerate=44100, 
                              channels=1, device=device_id)
            sd.wait()
            volume = np.max(np.abs(recording))
            print(f"Max volume: {volume}")
            return volume > 0.01
        else:
            print(f"Testing OUTPUT device {device_id}...")
            # Generate test tone
            samples = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float32)
            sd.play(samples, samplerate=44100, device=device_id)
            sd.wait()
            print("You should have heard a beep!")
            return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

def find_working_combination():
    print("\n=== FINDING WORKING COMBINATIONS ===")
    devices = sd.query_devices()
    
    # Test default first
    print("Testing default devices...")
    try:
        sd.check_input_settings()
        sd.check_output_settings()
        print("✅ Default devices work!")
        return None, None
    except:
        print("❌ Default devices failed")
    
    # Test individual devices
    working_inputs = []
    working_outputs = []
    
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            if test_device(i, is_input=True):
                working_inputs.append(i)
                print(f"✅ Input device {i} works")
        
        if dev['max_output_channels'] > 0:
            if test_device(i, is_input=False):
                working_outputs.append(i)
                print(f"✅ Output device {i} works")
    
    # Try combinations
    for inp in working_inputs:
        for out in working_outputs:
            try:
                print(f"Testing combination: Input {inp} -> Output {out}")
                def passthrough(indata, outdata, frames, time, status):
                    outdata[:] = indata
                
                with sd.Stream(device=(inp, out), channels=1, callback=passthrough):
                    sd.sleep(2000)
                
                print(f"🎉 SUCCESS: Input {inp} -> Output {out}")
                return inp, out
            except:
                continue
    
    print("😞 No working combination found")
    return None, None

if __name__ == "__main__":
    list_devices()
    input("\nPress Enter to test devices...")
    inp, out = find_working_combination()
    
    if inp is not None or out is not None:
        print(f"\n=== RECOMMENDED SETTINGS ===")
        print(f"Input device: {inp}")
        print(f"Output device: {out}")
        print("\nUse these values in the main app!")
    else:
        print("\nNo working audio configuration found.")
        print("Try installing VB-Cable virtual audio cable.")