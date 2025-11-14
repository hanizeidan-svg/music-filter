import sounddevice as sd
import numpy as np
import time

def test_ai_processing():
    """Test if AI processing is actually happening in real-time"""
    
    from audio_processor import AudioProcessor
    
    print("=== REAL-TIME AI PROCESSING TEST ===")
    print("This will test if the AI is actually processing live audio")
    print("You should hear the audio change when moving the slider!")
    
    processor = AudioProcessor()
    
    def callback(indata, outdata, frames, time_info, status):
        # Get mono audio
        audio_input = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        
        # Process with AI
        processed = processor.process_audio_chunk(audio_input)
        
        # Output
        outdata[:, 0] = processed[:frames]
        if outdata.shape[1] > 1:
            for ch in range(1, outdata.shape[1]):
                outdata[:, ch] = outdata[:, 0]
        
        # Print status occasionally
        if np.random.random() < 0.05:  # 5% chance
            volume = np.sqrt(np.mean(audio_input**2))
            if volume > 0.01:
                print(f"Audio level: {volume:.3f} - AI should be processing!")
    
    try:
        stream = sd.Stream(callback=callback, channels=1, samplerate=44100, blocksize=1024)
        stream.start()
        
        print("\n🎵 Stream started! Play some music and check the logs.")
        print("The AI should log 'APPLYING AI SEPARATION' every second.")
        print("\nPress Enter to stop...")
        input()
        
        stream.stop()
        stream.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ai_processing()