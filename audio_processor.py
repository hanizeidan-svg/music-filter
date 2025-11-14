import numpy as np
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import sounddevice as sd
import logging
import os
import time
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '1'

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.model = None
        self.sample_rate = 44100
        self.chunk_size = 44100  # 1 second for Demucs
        self.music_ratio = 0.5
        self.is_processing = False
        self.output_stream = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_processed_time = 0
        self.processing_interval = 1.0
        
        self.load_model()
    
    def load_model(self):
        """Load Demucs model"""
        try:
            logger.info("Loading AI model...")
            self.model = get_model('htdemucs')
            self.model.eval()
            logger.info("AI model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            self.model = None
            raise
    
    def ensure_stereo(self, audio_data):
        """Convert to stereo for Demucs"""
        if audio_data.ndim == 1:
            return np.column_stack((audio_data, audio_data))
        return audio_data
    
    def demucs_separation(self, audio_data):
        """Apply Demucs AI separation to audio data"""
        if self.model is None or len(audio_data) < 1000:
            return audio_data, audio_data
        
        try:
            # Ensure we have the right chunk size
            if len(audio_data) > self.chunk_size:
                process_data = audio_data[:self.chunk_size]
            else:
                process_data = audio_data
            
            # Convert to stereo
            stereo_audio = self.ensure_stereo(process_data)
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(stereo_audio.T).float()
            audio_tensor = audio_tensor.unsqueeze(0)
            
            # AI separation
            with torch.no_grad():
                sources = apply_model(self.model, audio_tensor, progress=False)
            
            # Extract vocals and music
            vocals = sources[0, 3].numpy()  # vocals source
            music = (sources[0, 0] + sources[0, 1] + sources[0, 2]).numpy() / 3.0  # combine instruments
            
            # Convert back to [samples, channels] and to mono
            vocals = np.mean(vocals.T, axis=1)
            music = np.mean(music.T, axis=1)
            
            logger.info(f"AI Separation - Vocals: {np.max(np.abs(vocals)):.3f}, Music: {np.max(np.abs(music)):.3f}")
            
            return vocals, music
            
        except Exception as e:
            logger.error(f"AI processing error: {e}")
            return audio_data, audio_data
    
    def process_audio_chunk(self, audio_data):
        """Main audio processing with AI separation and VOLUME BOOST"""
        current_time = time.time()
        
        # Check if we should process with AI
        should_process_ai = (
            len(audio_data) >= self.chunk_size and 
            (current_time - self.last_processed_time) >= self.processing_interval
        )
        
        if not should_process_ai:
            # Return original audio while waiting for AI processing
            return audio_data
        
        try:
            logger.info(">>> APPLYING AI SEPARATION...")
            
            # Get AI-separated vocals and music
            vocals, music = self.demucs_separation(audio_data)
            
            # Mix based on current ratio with STRONG separation and VOLUME BOOST
            if self.music_ratio < 0.1:
                # VOCAL MODE: 95% vocals, 5% music + VOLUME BOOST
                result = (vocals * 0.95 + music * 0.05) * 3.0  # 3x volume boost
                logger.info(">>> VOCAL EMPHASIS MODE - BOOSTED VOLUME")
            elif self.music_ratio > 0.9:
                # MUSIC MODE: 5% vocals, 95% music + VOLUME BOOST
                result = (vocals * 0.05 + music * 0.95) * 3.0  # 3x volume boost
                logger.info(">>> MUSIC EMPHASIS MODE - BOOSTED VOLUME")
            else:
                # BALANCED: Smooth transition + VOLUME BOOST
                result = ((vocals * (1 - self.music_ratio)) + (music * self.music_ratio)) * 2.0  # 2x volume boost
                logger.info(f">>> BALANCED MODE (ratio: {self.music_ratio:.1f}) - BOOSTED VOLUME")
            
            # Ensure proper length
            if len(result) > len(audio_data):
                result = result[:len(audio_data)]
            elif len(result) < len(audio_data):
                # Pad with original audio if AI output is shorter
                padded = np.zeros_like(audio_data)
                padded[:len(result)] = result
                result = padded
            
            # Prevent clipping
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result = result / max_val
                logger.info(f">>> Normalized to prevent clipping")
            
            # Update timing
            self.last_processed_time = current_time
            
            # Log the effect
            original_max = np.max(np.abs(audio_data))
            processed_max = np.max(np.abs(result))
            logger.info(f">>> AI Processing Complete - Original: {original_max:.3f}, Processed: {processed_max:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            return audio_data
    
    def start_processing(self, input_device=None, output_device=None):
        """Start real-time audio processing"""
        if self.is_processing:
            return
        
        if self.model is None:
            raise RuntimeError("AI model not loaded")
        
        self.is_processing = True
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_processed_time = time.time()
        
        def audio_callback(indata, outdata, frames, time_info, status):
            if status:
                if status.input_overflow:
                    logger.debug("Input overflow")
                if status.output_underflow:
                    logger.debug("Output underflow")
            
            if not self.is_processing:
                outdata.fill(0)
                return
            
            try:
                # Get audio input (convert to mono)
                if indata.ndim > 1:
                    audio_input = indata[:, 0]
                else:
                    audio_input = indata.flatten()
                
                # Add to growing buffer
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_input])
                
                # Keep buffer manageable (max 3 seconds)
                max_buffer_size = 3 * self.sample_rate
                if len(self.audio_buffer) > max_buffer_size:
                    self.audio_buffer = self.audio_buffer[-max_buffer_size:]
                
                # Process with AI separation
                processed_audio = self.process_audio_chunk(self.audio_buffer)
                
                # Output the most recent portion
                output_start = max(0, len(processed_audio) - frames)
                output_end = min(len(processed_audio), output_start + frames)
                
                if output_end > output_start:
                    outdata[:output_end-output_start, 0] = processed_audio[output_start:output_end]
                
                # Fill any remaining with zeros
                if output_end - output_start < frames:
                    outdata[output_end-output_start:, 0] = 0
                
                # Copy to other channels
                if outdata.shape[1] > 1:
                    for ch in range(1, outdata.shape[1]):
                        outdata[:, ch] = outdata[:, 0]
                        
                # Log occasionally
                if np.random.random() < 0.01:  # 1% chance to log
                    volume = np.sqrt(np.mean(audio_input**2))
                    if volume > 0.01:
                        logger.info(f">>> Audio level: {volume:.3f}")
                        
            except Exception as e:
                logger.error(f"Callback error: {e}")
                outdata.fill(0)
        
        try:
            # Start audio stream
            self.output_stream = sd.Stream(
                callback=audio_callback,
                samplerate=self.sample_rate,
                blocksize=2048,
                device=(None, None),
                channels=1,
                dtype=np.float32,
                latency='low'
            )
            
            self.output_stream.start()
            logger.info(">>> REAL-TIME AI PROCESSING STARTED!")
            logger.info(">>> Play music and move slider to hear vocal/music separation!")
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.is_processing = False
            raise
    
    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        if self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None
                logger.info(">>> Processing stopped")
            except Exception as e:
                logger.error(f"Stop error: {e}")
    
    def set_music_ratio(self, ratio):
        """Set separation ratio"""
        old_ratio = self.music_ratio
        self.music_ratio = max(0.0, min(1.0, ratio))
        
        # Log mode changes
        if abs(old_ratio - self.music_ratio) > 0.2:
            if self.music_ratio < 0.2:
                logger.info(">>> SWITCHED TO VOCAL MODE")
            elif self.music_ratio > 0.8:
                logger.info(">>> SWITCHED TO MUSIC MODE")
            else:
                logger.info(">>> SWITCHED TO BALANCED MODE")