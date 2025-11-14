import sounddevice as sd
import numpy as np

def test_device_combinations():
    print("Testing audio device combinations...")
    
    devices = sd.query_devices()
    input_devices = []
    output_devices = []
    
    # List all devices
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            input_devices.append((i, dev['name']))
        if dev['max_output_channels'] > 0:
            output_devices.append((i, dev['name']))
    
    print("\nInput Devices:")
    for idx, name in input_devices:
        print(f"  {idx}: {name}")
    
    print("\nOutput Devices:")
    for idx, name in output_devices:
        print(f"  {idx}: {name}")
    
    # Test combinations
    print("\nTesting combinations...")
    for input_idx, input_name in input_devices:
        if 'stereo mix' in input_name.lower():
            for output_idx, output_name in output_devices:
                try:
                    duration = 2
                    print(f"Testing: {input_name} -> {output_name}")
                    
                    def callback(indata, outdata, frames, time, status):
                        outdata[:] = indata
                    
                    with sd.Stream(device=(input_idx, output_idx), 
                                 channels=1, callback=callback):
                        sd.sleep(duration * 1000)
                    
                    print(f"✅ SUCCESS: {input_name} -> {output_name}")
                    return input_idx, output_idx
                    
                except Exception as e:
                    print(f"❌ FAILED: {e}")
    
    return None, None

if __name__ == "__main__":
    input_dev, output_dev = test_device_combinations()
    if input_dev is not None:
        print(f"\n🎉 Working combination found!")
        print(f"Input: {input_dev}, Output: {output_dev}")
    else:
        print("\n😞 No working combination found. Try virtual audio cable.")