import numpy as np
import torch
from demucs.pretrained import get_model
from demucs.apply import apply_model
import sounddevice as sd
import logging
import os
import time
import warnings
import threading
from queue import Queue

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
        
        # Manual device selection
        self.input_device = None
        self.output_device = None
        
        # Processing mode
        self.processing_mode = "transform"
        
        # Audio buffers and synchronization
        self.audio_buffer = np.array([], dtype=np.float32)
        self.processed_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.last_ai_time = 0
        self.ai_processing_interval = 2.0  # Process AI every 2 seconds
        
        # For real-time processing
        self.current_output_chunk = None
        self.output_position = 0
        
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
    
    def apply_ai_separation(self, audio_chunk):
        """Apply AI separation to an audio chunk - SIMPLIFIED AND FIXED"""
        if self.model is None or len(audio_chunk) < 1000:
            return audio_chunk
        
        try:
            logger.info(f">>> AI Processing {len(audio_chunk)} samples...")
            
            # Ensure proper length for Demucs
            target_length = min(len(audio_chunk), self.chunk_size)
            process_data = audio_chunk[:target_length]
            
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
            
            # Convert back to mono
            vocals_mono = np.mean(vocals.T, axis=1)
            music_mono = np.mean(music.T, axis=1)
            
            logger.info(f"AI Results - Vocals: {np.max(np.abs(vocals_mono)):.3f}, Music: {np.max(np.abs(music_mono)):.3f}")
            
            # Apply STRONG separation based on ratio
            if self.music_ratio < 0.1:
                # VOCAL MODE: 98% vocals, 2% music
                result = vocals_mono * 0.98 + music_mono * 0.02
                logger.info(">>> STRONG VOCAL EMPHASIS APPLIED")
            elif self.music_ratio > 0.9:
                # MUSIC MODE: 2% vocals, 98% music
                result = vocals_mono * 0.02 + music_mono * 0.98
                logger.info(">>> STRONG MUSIC EMPHASIS APPLIED")
            else:
                # BALANCED: Linear mix
                result = (vocals_mono * (1 - self.music_ratio)) + (music_mono * self.music_ratio)
                logger.info(f">>> BALANCED MIX (ratio: {self.music_ratio:.1f})")
            
            # Volume normalization
            original_level = np.max(np.abs(process_data))
            processed_level = np.max(np.abs(result))
            
            if processed_level > 0:
                # Maintain similar volume level
                result = result * (original_level / processed_level) * 0.8
            
            logger.info(f">>> Volume - Original: {original_level:.3f}, Processed: {np.max(np.abs(result)):.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"AI processing error: {e}")
            return audio_chunk
    
    def process_audio_realtime(self):
        """Main processing loop - called from input callback"""
        current_time = time.time()
        
        # Check if we should process with AI
        should_process_ai = (
            self.processing_mode == "transform" and
            len(self.audio_buffer) >= self.chunk_size and
            (current_time - self.last_ai_time) >= self.ai_processing_interval
        )
        
        if should_process_ai:
            try:
                with self.buffer_lock:
                    # Take a chunk for AI processing
                    ai_chunk = self.audio_buffer[:self.chunk_size].copy()
                    self.audio_buffer = self.audio_buffer[self.chunk_size:]
                
                # Process with AI
                processed_chunk = self.apply_ai_separation(ai_chunk)
                
                # Replace the corresponding section in processed buffer
                with self.buffer_lock:
                    # Ensure we have a processed buffer
                    if len(self.processed_buffer) < len(ai_chunk):
                        self.processed_buffer = np.zeros_like(self.audio_buffer)
                    
                    # Replace the processed section
                    replace_length = min(len(processed_chunk), len(self.processed_buffer))
                    self.processed_buffer[:replace_length] = processed_chunk[:replace_length]
                
                self.last_ai_time = current_time
                logger.info(f">>> AI processing completed - buffer updated")
                
            except Exception as e:
                logger.error(f"Real-time processing error: {e}")
    
    def start_processing(self):
        """Start audio processing with FIXED real-time handling"""
        if self.is_processing:
            return
        
        if self.model is None and self.processing_mode == "transform":
            raise RuntimeError("AI model not loaded")
        
        if self.input_device is None or self.output_device is None:
            raise RuntimeError("Input and output devices must be selected first")
        
        self.is_processing = True
        self.audio_buffer = np.array([], dtype=np.float32)
        self.processed_buffer = np.array([], dtype=np.float32)
        self.last_ai_time = time.time()
        
        def input_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Input status: {status}")
            
            if self.is_processing:
                try:
                    # Get audio input (convert to mono)
                    if indata.ndim > 1:
                        audio_input = indata[:, 0]  # Use first channel
                    else:
                        audio_input = indata.flatten()
                    
                    # Add to buffer with lock
                    with self.buffer_lock:
                        self.audio_buffer = np.concatenate([self.audio_buffer, audio_input])
                        
                        # For pass-through mode, keep processed buffer in sync
                        if self.processing_mode == "pass_through":
                            self.processed_buffer = self.audio_buffer.copy()
                    
                    # Keep buffer manageable (max 3 seconds)
                    max_buffer_size = 3 * self.sample_rate
                    with self.buffer_lock:
                        if len(self.audio_buffer) > max_buffer_size:
                            self.audio_buffer = self.audio_buffer[-max_buffer_size:]
                        if len(self.processed_buffer) > max_buffer_size:
                            self.processed_buffer = self.processed_buffer[-max_buffer_size:]
                    
                    # Process audio if in transform mode
                    if self.processing_mode == "transform":
                        self.process_audio_realtime()
                    
                    # Log occasionally
                    if np.random.random() < 0.02:  # 2% chance to log
                        volume = np.sqrt(np.mean(audio_input**2))
                        if volume > 0.01:
                            mode = "PASS-THROUGH" if self.processing_mode == "pass_through" else "AI TRANSFORM"
                            with self.buffer_lock:
                                buffer_size = len(self.audio_buffer)
                            logger.info(f">>> Input: {volume:.3f} | Mode: {mode} | Buffer: {buffer_size}")
                            
                except Exception as e:
                    logger.error(f"Input callback error: {e}")
        
        def output_callback(outdata, frames, time_info, status):
            if status:
                logger.debug(f"Output status: {status}")
            
            if not self.is_processing:
                outdata.fill(0)
                return
            
            try:
                with self.buffer_lock:
                    # Determine which buffer to use
                    if self.processing_mode == "pass_through" or len(self.processed_buffer) < frames:
                        # Use original audio buffer for pass-through or if no processed data
                        source_buffer = self.audio_buffer
                    else:
                        # Use processed buffer for AI transform
                        source_buffer = self.processed_buffer
                
                # Output audio
                if len(source_buffer) >= frames:
                    outdata[:, 0] = source_buffer[:frames]
                    
                    # Remove used audio from buffers
                    with self.buffer_lock:
                        self.audio_buffer = self.audio_buffer[frames:]
                        if len(self.processed_buffer) >= frames:
                            self.processed_buffer = self.processed_buffer[frames:]
                        else:
                            self.processed_buffer = np.array([], dtype=np.float32)
                else:
                    # Not enough data, output silence
                    outdata.fill(0)
                    logger.debug("Buffer underflow - outputting silence")
                
                # Copy to other channels
                if outdata.shape[1] > 1:
                    for ch in range(1, outdata.shape[1]):
                        outdata[:, ch] = outdata[:, 0]
                        
            except Exception as e:
                logger.error(f"Output callback error: {e}")
                outdata.fill(0)
        
        try:
            # Get device info for logging
            devices = sd.query_devices()
            input_name = devices[self.input_device]['name'] if self.input_device < len(devices) else f"Device {self.input_device}"
            output_name = devices[self.output_device]['name'] if self.output_device < len(devices) else f"Device {self.output_device}"
            
            mode_text = "PASS-THROUGH" if self.processing_mode == "pass_through" else "AI TRANSFORM"
            logger.info(f"Starting streams - Mode: {mode_text}")
            logger.info(f"Input: {input_name} (Device {self.input_device})")
            logger.info(f"Output: {output_name} (Device {self.output_device})")
            
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
                callback=output_callback,
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
            
            logger.info(f">>> {mode_text} MODE STARTED SUCCESSFULLY!")
            
            if self.processing_mode == "pass_through":
                logger.info(">>> Audio passing through unchanged - verify routing")
            else:
                logger.info(">>> AI separation active - move slider for vocal/music control!")
            
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
        
        # Clear buffers
        self.audio_buffer = np.array([], dtype=np.float32)
        self.processed_buffer = np.array([], dtype=np.float32)
        
        logger.info(">>> Processing stopped")
    
    def set_music_ratio(self, ratio):
        """Set separation ratio"""
        old_ratio = self.music_ratio
        self.music_ratio = max(0.0, min(1.0, ratio))
        
        # Log mode changes
        if abs(old_ratio - self.music_ratio) > 0.1:
            if self.music_ratio < 0.1:
                logger.info(">>> SWITCHED TO VOCAL MODE - Expecting mostly vocals")
            elif self.music_ratio > 0.9:
                logger.info(">>> SWITCHED TO MUSIC MODE - Expecting mostly music")
            else:
                logger.info(f">>> SWITCHED TO BALANCED MODE - Ratio: {self.music_ratio:.1f}")
