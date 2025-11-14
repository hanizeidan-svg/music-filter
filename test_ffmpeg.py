import torch
import torchaudio
import subprocess
import sys
import os

def test_ffmpeg():
    print("=== Testing FFmpeg Installation ===")
    
    # Test system FFmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ System FFmpeg found:")
            print(result.stdout.split('\n')[0])
        else:
            print("❌ System FFmpeg not found")
    except:
        print("❌ System FFmpeg not accessible")
    
    # Test torchaudio
    print("\n=== Testing Torchaudio ===")
    try:
        print(f"Torchaudio version: {torchaudio.__version__}")
        print(f"Torch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Test backend
        backend = torchaudio.list_audio_backends()
        print(f"Available backends: {backend}")
        
        # Test loading a simple file
        print("✅ Torchaudio imported successfully")
    except Exception as e:
        print(f"❌ Torchaudio error: {e}")
    
    # Test Demucs
    print("\n=== Testing Demucs ===")
    try:
        from demucs.pretrained import get_model
        model = get_model('htdemucs')
        print("✅ Demucs model loaded successfully")
    except Exception as e:
        print(f"❌ Demucs error: {e}")

if __name__ == "__main__":
    test_ffmpeg()