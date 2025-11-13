import numpy as np
import torch
import torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
import sounddevice as sd
import threading
import queue
import time
import logging

# Set up logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            logger.info("Loading Demucs model...")
            self.model = get_model('htdemucs')
            self.model.eval()
            logger.info("Model loaded successfully")
            
            # Check if model downloaded
            import os
            from pathlib import Path
            home = Path.home()
            demucs_path = home / '.cache' / 'torch' / 'hub' / 'checkpoints'
            if demucs_path.exists():
                logger.info(f"Model cache path: {demucs_path}")
                files = list(demucs_path.glob('*'))
                logger.info(f"Found {len(files)} files in cache")
                for f in files:
                    logger.info(f"  - {f.name}")
                    
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def ensure_stereo(self, audio_data):
        """Convert mono audio to stereo by duplicating the channel"""
        if audio_data.ndim == 1:
            # Mono to stereo: duplicate the channel
            return np.column_stack((audio_data, audio_data))
        elif audio_data.ndim == 2 and audio_data.shape[1] == 1:
            # Single channel 2D array to stereo
            return np.column_stack((audio_data[:, 0], audio_data[:, 0]))
        else:
            # Already stereo or multi-channel
            return audio_data
    
    def process_audio_chunk(self, audio_data):
        """Process a chunk of audio data with music/vocal separation"""
        if self.model is None:
            logger.warning("Model not available, using fallback processing")
            return self.fallback_processing(audio_data)
        
        try:
            # Ensure we have stereo audio
            stereo_audio = self.ensure_stereo(audio_data)
            
            # Convert to tensor for Demucs - shape should be [samples, 2] for stereo
            audio_tensor = torch.from_numpy(stereo_audio.T).float()  # Transpose to [channels, samples]
            
            # Add batch dimension: [1, channels, samples]
            audio_tensor = audio_tensor.unsqueeze(0)
            
            logger.debug(f"Audio tensor shape: {audio_tensor.shape}")
            
            # Separate sources (vocals, drums, bass, other)
            with torch.no_grad():
                sources = apply_model(self.model, audio_tensor, progress=False)
            
            # sources shape: [batch, sources, channels, samples]
            # Sources order: ['drums', 'bass', 'other', 'vocals']
            vocals = sources[0, 3].numpy()  # vocals source
            music = np.mean(sources[0, 0:3].numpy(), axis=0)  # combine drums, bass, other
            
            # Mix based on music ratio
            mixed_audio = (vocals * (1 - self.music_ratio)) + (music * self.music_ratio)
            
            # Convert back to mono if input was mono
            if audio_data.ndim == 1 or (audio_data.ndim == 2 and audio_data.shape[1] == 1):
                mixed_audio = np.mean(mixed_audio, axis=0)  # Average channels to mono
            
            return mixed_audio
            
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            return self.fallback_processing(audio_data)
    
    def fallback_processing(self, audio_data):
        """Fallback processing when model isn't available or fails"""
        logger.info("Using fallback processing")
        
        # Simple high-pass/low-pass filter simulation based on music ratio
        if len(audio_data) > 10:  # Ensure we have enough samples
            # For vocals (low ratio): emphasize higher frequencies
            # For music (high ratio): emphasize lower frequencies
            if self.music_ratio < 0.3:
                # Vocals mode: reduce bass
                processed = audio_data * 0.8  # Simple attenuation
            elif self.music_ratio > 0.7:
                # Music mode: reduce treble
                processed = audio_data * 1.2  # Simple boost
            else:
                # Balanced: minimal processing
                processed = audio_data
            return processed
        return audio_data
    
    def start_processing(self, input_device=None, output_device=None):
        """Start real-time audio processing"""
        if self.is_processing:
            return
        
        self.is_processing = True
        
        def audio_callback(indata, outdata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            if self.is_processing:
                try:
                    # Process the incoming audio (use first channel if multi-channel)
                    if indata.ndim > 1:
                        audio_input = indata[:, 0]  # Use first channel
                    else:
                        audio_input = indata
                    
                    processed_audio = self.process_audio_chunk(audio_input)
                    
                    # Ensure output matches expected shape
                    if outdata.ndim > 1:
                        # Multi-channel output
                        for channel in range(outdata.shape[1]):
                            if len(processed_audio) == len(outdata):
                                outdata[:, channel] = processed_audio
                            else:
                                outdata.fill(0)
                    else:
                        # Mono output
                        if len(processed_audio) == len(outdata):
                            outdata[:] = processed_audio
                        else:
                            outdata.fill(0)
                            
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")
                    outdata.fill(0)
        
        try:
            # Start audio stream - use 2 channels for input to get stereo
            self.output_stream = sd.Stream(
                callback=audio_callback,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=(input_device, output_device),
                channels=2,  # Request stereo input
                dtype=np.float32
            )
            
            self.output_stream.start()
            logger.info("Audio processing started")
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.is_processing = False
    
    def stop_processing(self):
        """Stop audio processing"""
        self.is_processing = False
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
        logger.info("Audio processing stopped")
    
    def set_music_ratio(self, ratio):
        """Set the music/vocal ratio (0.0 = vocals only, 1.0 = music only)"""
        self.music_ratio = max(0.0, min(1.0, ratio))  # Clamp between 0 and 1
        logger.info(f"Music ratio set to: {self.music_ratio}")