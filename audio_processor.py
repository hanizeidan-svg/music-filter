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
        self.input_stream = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_processed_time = 0
        self.processing_interval = 1.0
        
        # Manual device selection - will be set by UI
        self.input_device = None
        self.output_device = None
        
        # Processing mode
        self.processing_mode = "transform"  # "transform" or "pass_through"
        
        self.load_model()
    
    def set_processing_mode(self, mode):
        """Set processing mode: 'transform' or 'pass_through'"""
        self.processing_mode = mode
        logger.info(f"Processing mode set to: {mode}")
    
    def get_audio_devices(self):
        """Get list of all audio devices with their capabilities"""
        devices = []
        try:
            all_devices = sd.query_devices()
            for i, device in enumerate(all_devices):
                device_info = {
                    'index': i,
                    'name': device['name'],
                    'max_input_channels': device['max_input_channels'],
                    'max_output_channels': device['max_output_channels'],
                    'default_samplerate': device.get('default_samplerate', 44100),
                    'hostapi': device.get('hostapi', 0)
                }
                devices.append(device_info)
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
        
        return devices
    
    def set_devices(self, input_device, output_device):
        """Set input and output devices manually"""
        self.input_device = input_device
        self.output_device = output_device
        logger.info(f"Devices set - Input: {input_device}, Output: {output_device}")
    
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
        """Main audio processing - either AI transformation or pass-through"""
        if self.processing_mode == "pass_through":
            # Simple pass-through - no processing
            return audio_data
        
        # AI Transformation mode
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
                result = (vocals * 0.95 + music * 0.05) * 2.0  # 2x volume boost
                logger.info(">>> VOCAL EMPHASIS MODE")
            elif self.music_ratio > 0.9:
                # MUSIC MODE: 5% vocals, 95% music + VOLUME BOOST
                result = (vocals * 0.05 + music * 0.95) * 2.0  # 2x volume boost
                logger.info(">>> MUSIC EMPHASIS MODE")
            else:
                # BALANCED: Smooth transition + VOLUME BOOST
                result = ((vocals * (1 - self.music_ratio)) + (music * self.music_ratio)) * 1.5  # 1.5x volume boost
                logger.info(f">>> BALANCED MODE (ratio: {self.music_ratio:.1f})")
            
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
                logger.info(">>> Normalized to prevent clipping")
            
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
    
    def start_processing(self):
        """Start audio processing with manually selected devices"""
        if self.is_processing:
            return
        
        if self.model is None and self.processing_mode == "transform":
            raise RuntimeError("AI model not loaded")
        
        if self.input_device is None or self.output_device is None:
            raise RuntimeError("Input and output devices must be selected first")
        
        self.is_processing = True
        self.audio_buffer = np.array([], dtype=np.float32)
        self.last_processed_time = time.time()
        
        # Audio processing callback
        def audio_callback(outdata, frames, time_info, status):
            if status:
                if status.output_underflow:
                    logger.debug("Output underflow")
            
            if not self.is_processing:
                outdata.fill(0)
                return
            
            try:
                # Get audio from buffer (filled by input stream)
                if len(self.audio_buffer) >= frames:
                    # We have enough data, process and output
                    chunk_to_process = self.audio_buffer[:frames]
                    self.audio_buffer = self.audio_buffer[frames:]
                    
                    # Process audio (either transform or pass-through)
                    processed_audio = self.process_audio_chunk(chunk_to_process)
                    
                    # Output processed audio
                    outdata[:, 0] = processed_audio[:frames]
                    
                    # Copy to other channels
                    if outdata.shape[1] > 1:
                        for ch in range(1, outdata.shape[1]):
                            outdata[:, ch] = outdata[:, 0]
                            
                else:
                    # Not enough data, output silence
                    outdata.fill(0)
                    logger.debug("Buffer underflow - outputting silence")
                        
            except Exception as e:
                logger.error(f"Output callback error: {e}")
                outdata.fill(0)
        
        # Input callback to capture from input device
        def input_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Input status: {status}")
            
            if self.is_processing:
                # Add incoming audio to buffer
                if indata.ndim > 1:
                    audio_input = indata[:, 0]  # Use first channel
                else:
                    audio_input = indata.flatten()
                
                self.audio_buffer = np.concatenate([self.audio_buffer, audio_input])
                
                # Keep buffer manageable (max 5 seconds)
                max_buffer_size = 5 * self.sample_rate
                if len(self.audio_buffer) > max_buffer_size:
                    self.audio_buffer = self.audio_buffer[-max_buffer_size:]
                
                # Log buffer status occasionally
                if len(self.audio_buffer) > self.sample_rate and np.random.random() < 0.01:
                    volume = np.sqrt(np.mean(audio_input**2))
                    mode = "PASS-THROUGH" if self.processing_mode == "pass_through" else "AI TRANSFORM"
                    if volume > 0.01:
                        logger.info(f">>> Input audio: {volume:.3f} | Mode: {mode} | Buffer: {len(self.audio_buffer)} samples")
        
        try:
            # Get device info for logging
            devices = sd.query_devices()
            input_name = devices[self.input_device]['name'] if self.input_device < len(devices) else f"Device {self.input_device}"
            output_name = devices[self.output_device]['name'] if self.output_device < len(devices) else f"Device {self.output_device}"
            
            mode_text = "PASS-THROUGH" if self.processing_mode == "pass_through" else "AI TRANSFORM"
            logger.info(f"Starting streams - Mode: {mode_text}, Input: {input_name}, Output: {output_name}")
            
            # Start INPUT stream
            self.input_stream = sd.InputStream(
                callback=input_callback,
                samplerate=self.sample_rate,
                blocksize=1024,
                device=self.input_device,
                channels=1,
                dtype=np.float32
            )
            
            # Start OUTPUT stream
            self.output_stream = sd.OutputStream(
                callback=audio_callback,
                samplerate=self.sample_rate,
                blocksize=1024,
                device=self.output_device,
                channels=1,
                dtype=np.float32,
                latency='low'
            )
            
            # Start both streams
            self.input_stream.start()
            self.output_stream.start()
            
            mode_text = "PASS-THROUGH" if self.processing_mode == "pass_through" else "AI TRANSFORM"
            logger.info(f">>> {mode_text} MODE STARTED!")
            logger.info(f">>> Input: {input_name} (Device {self.input_device})")
            logger.info(f">>> Output: {output_name} (Device {self.output_device})")
            
            if self.processing_mode == "pass_through":
                logger.info(">>> Audio is passing through unchanged - verify routing works")
            else:
                logger.info(">>> Play audio and move slider for AI separation!")
            
        except Exception as e:
            logger.error(f"Error starting audio streams: {e}")
            self.stop_processing()
            raise
    
    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        
        if self.input_stream:
            try:
                self.input_stream.stop()
                self.input_stream.close()
                self.input_stream = None
            except Exception as e:
                logger.error(f"Error stopping input stream: {e}")
        
        if self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None
            except Exception as e:
                logger.error(f"Error stopping output stream: {e}")
        
        logger.info(">>> Processing stopped")
    
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