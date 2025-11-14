import numpy as np
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import sounddevice as sd
import time

def test_demucs_processing():
    print("=== TESTING DEMUCS AI PROCESSING ===")
    
    # Load model
    print("Loading Demucs model...")
    model = get_model('htdemucs')
    model.eval()
    print("✅ Model loaded")
    
    # Create test audio (1 second of sine wave)
    duration = 1.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Mixed signal: 440Hz (voice-like) + 220Hz (music-like)
    voice_signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    music_signal = 0.5 * np.sin(2 * np.pi * 220 * t)
    mixed_signal = voice_signal + music_signal
    
    print(f"Created test signal: {len(mixed_signal)} samples")
    
    # Convert to stereo for Demucs
    stereo_audio = np.column_stack((mixed_signal, mixed_signal))
    
    # Convert to tensor
    audio_tensor = torch.from_numpy(stereo_audio.T).float().unsqueeze(0)
    print(f"Input tensor shape: {audio_tensor.shape}")
    
    # Process with Demucs
    print("Processing with AI...")
    start_time = time.time()
    
    with torch.no_grad():
        sources = apply_model(model, audio_tensor, progress=False)
    
    processing_time = time.time() - start_time
    print(f"✅ AI processing completed in {processing_time:.2f} seconds")
    print(f"Output sources shape: {sources.shape}")
    
    # Show separation results
    vocals = sources[0, 3].numpy()  # vocals
    music = sources[0, 0].numpy()   # drums (part of music)
    
    print(f"\n=== SEPARATION RESULTS ===")
    print(f"Original signal range: {np.min(mixed_signal):.3f} to {np.max(mixed_signal):.3f}")
    print(f"Vocals range: {np.min(vocals):.3f} to {np.max(vocals):.3f}")
    print(f"Music range: {np.min(music):.3f} to {np.max(music):.3f}")
    
    # Test mixing
    print(f"\n=== MIXING TEST ===")
    for ratio in [0.0, 0.5, 1.0]:
        mixed = (vocals * (1 - ratio)) + (music * ratio)
        mixed = np.mean(mixed, axis=0)  # to mono
        print(f"Ratio {ratio:.1f}: range {np.min(mixed):.3f} to {np.max(mixed):.3f}")

if __name__ == "__main__":
    test_demucs_processing()