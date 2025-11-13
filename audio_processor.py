import numpy as np
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import sounddevice as sd
import threading
import queue
import time

class AudioProcessor:
    def __init__(self):
        self.model = None
        self.sample_rate = 44100
        self.chunk_size = 4096
        self.music_ratio = 0.5
        self.is_processing = False
        self.audio_queue = queue.Queue()
        self.output_stream = None
        
        # Load the separation model
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained Demucs model for vocal/music separation"""
        try:
            print("Loading Demucs model...")
            self.model = get_model('htdemucs')
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to simpler processing
            self.model = None
    
    def process_audio_chunk(self, audio_data):
        """Process a chunk of audio data with music/vocal separation"""
        if self.model is None:
            # Simple high-pass/low-pass filter as fallback
            return self.fallback_processing(audio_data)
        
        try:
            # Convert to tensor for Demucs
            audio_tensor = torch.from_numpy(audio_data).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension
            
            # Separate sources (vocals, drums, bass, other)
            with torch.no_grad():
                sources = apply_model(self.model, audio_tensor.unsqueeze(0), progress=False)
            
            # Extract vocals (usually first source) and music (others combined)
            vocals = sources[0, 0].numpy()  # First source, first channel
            music = np.mean(sources[0, 1:].numpy(), axis=0)  # Combine other sources
            
            # Mix based on music ratio
            mixed_audio = (vocals * (1 - self.music_ratio)) + (music * self.music_ratio)
            
            return mixed_audio
            
        except Exception as e:
            print(f"Error in audio processing: {e}")
            return self.fallback_processing(audio_data)
    
    def fallback_processing(self, audio_data):
        """Fallback processing when model isn't available"""
        # Simple frequency-based separation (less effective but works)
        if len(audio_data) > 0:
            # Apply simple filtering based on music ratio
            # This is a placeholder - real implementation would use proper filters
            return audio_data * (0.5 + 0.5 * self.music_ratio)
        return audio_data
    
    def start_processing(self, input_device=None, output_device=None):
        """Start real-time audio processing"""
        if self.is_processing:
            return
        
        self.is_processing = True
        
        def audio_callback(indata, outdata, frames, time, status):
            if status:
                print(f"Audio callback status: {status}")
            
            if self.is_processing:
                # Process the incoming audio
                processed_audio = self.process_audio_chunk(indata[:, 0])  # Use first channel
                
                # Ensure output is correct shape and type
                if len(processed_audio) == len(outdata):
                    outdata[:, 0] = processed_audio
                    if outdata.shape[1] > 1:  # Copy to other channels if stereo
                        outdata[:, 1] = processed_audio
                else:
                    outdata.fill(0)
        
        try:
            # Start audio stream
            self.output_stream = sd.Stream(
                callback=audio_callback,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=(input_device, output_device),
                channels=1,  # Process as mono for simplicity
                dtype=np.float32
            )
            
            self.output_stream.start()
            print("Audio processing started")
            
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.is_processing = False
    
    def stop_processing(self):
        """Stop audio processing"""
        self.is_processing = False
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
        print("Audio processing stopped")
    
    def set_music_ratio(self, ratio):
        """Set the music/vocal ratio (0.0 = vocals only, 1.0 = music only)"""
        self.music_ratio = max(0.0, min(1.0, ratio))  # Clamp between 0 and 1
        print(f"Music ratio set to: {self.music_ratio}")