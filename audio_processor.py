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
from scipy import signal
import queue

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '1'

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.model = None
        self.sample_rate = 44100
        self.chunk_size = 22050  # 0.5 second for faster processing
        self.music_ratio = 0.5
        self.is_processing = False
        self.output_stream = None
        self.input_stream = None
        
        # Manual device selection
        self.input_device = None
        self.output_device = None
        
        # Processing mode
        self.processing_mode = "pass_through"
        
        # Audio buffers and threading
        self.audio_buffer = np.array([], dtype=np.float32)
        self.processed_audio = None
        self.buffer_lock = threading.Lock()
        
        # AI processing thread
        self.ai_thread = None
        self.ai_stop_event = threading.Event()
        self.ai_ready = False
        self.last_ai_time = 0
        self.ai_processing_interval = 3.0  # Process every 3 seconds
        
        self.load_model()
    
    def set_processing_mode(self, mode):
        """Set processing mode"""
        self.processing_mode = mode
        logger.info(f"Processing mode set to: {mode}")
        
        # Reset AI state when switching to AI mode
        if mode == "ai_transform":
            self.ai_ready = False
            self.processed_audio = None
    
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
    
    # SIMPLE TEST TRANSFORMATIONS - GUARANTEED TO WORK
    def apply_volume_reduction(self, audio_data):
        """Reduce volume by 90% - VERY OBVIOUS EFFECT"""
        return audio_data * 0.1
    
    def apply_volume_boost(self, audio_data):
        """Boost volume by 200% - VERY OBVIOUS EFFECT"""
        boosted = audio_data * 2.0
        max_val = np.max(np.abs(boosted))
        if max_val > 1.0:
            boosted = boosted / max_val
        return boosted
    
    def apply_robot_effect(self, audio_data):
        """Apply robot voice effect - VERY OBVIOUS EFFECT"""
        if len(audio_data) > 100:
            # Simple distortion + pitch effect
            distorted = np.tanh(audio_data * 3.0) * 0.7
            if len(distorted) > 1000:
                delay_samples = 30
                delayed = np.roll(distorted, delay_samples)
                delayed[:delay_samples] = 0
                return (distorted + delayed * 0.3) * 0.8
        return audio_data
    
    def apply_radio_effect(self, audio_data):
        """Apply AM radio effect - VERY OBVIOUS EFFECT"""
        if len(audio_data) > 100:
            nyquist = self.sample_rate / 2
            low = 400 / nyquist
            high = 2500 / nyquist
            b, a = signal.butter(3, [low, high], btype='band')
            radio_audio = signal.lfilter(b, a, audio_data)
            return np.tanh(radio_audio * 2.0) * 0.6
        return audio_data
    
    def ai_separation_worker(self):
        """Background worker for AI processing - NON-BLOCKING"""
        logger.info("AI processing worker started")
        
        while not self.ai_stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check if we should process
                should_process = (
                    self.processing_mode == "ai_transform" and
                    (current_time - self.last_ai_time) >= self.ai_processing_interval
                )
                
                if should_process:
                    with self.buffer_lock:
                        if len(self.audio_buffer) >= self.chunk_size:
                            # Take a chunk for processing
                            ai_chunk = self.audio_buffer[:self.chunk_size].copy()
                        else:
                            ai_chunk = None
                    
                    if ai_chunk is not None and self.model is not None:
                        logger.info(">>> AI Processing starting...")
                        start_time = time.time()
                        
                        try:
                            # Convert to stereo
                            if ai_chunk.ndim == 1:
                                stereo_audio = np.column_stack((ai_chunk, ai_chunk))
                            else:
                                stereo_audio = ai_chunk
                            
                            # Convert to tensor
                            audio_tensor = torch.from_numpy(stereo_audio.T).float()
                            audio_tensor = audio_tensor.unsqueeze(0)
                            
                            # AI separation
                            with torch.no_grad():
                                sources = apply_model(self.model, audio_tensor, progress=False)
                            
                            # Extract and mix
                            vocals = sources[0, 3].numpy()
                            music = (sources[0, 0] + sources[0, 1] + sources[0, 2]).numpy() / 3.0
                            
                            vocals_mono = np.mean(vocals.T, axis=1)
                            music_mono = np.mean(music.T, axis=1)
                            
                            # Apply separation
                            if self.music_ratio < 0.1:
                                result = vocals_mono * 0.9 + music_mono * 0.1
                                logger.info(">>> AI: VOCAL EMPHASIS")
                            elif self.music_ratio > 0.9:
                                result = vocals_mono * 0.1 + music_mono * 0.9
                                logger.info(">>> AI: MUSIC EMPHASIS")
                            else:
                                result = (vocals_mono * (1 - self.music_ratio)) + (music_mono * self.music_ratio)
                                logger.info(f">>> AI: BALANCED (ratio: {self.music_ratio:.1f})")
                            
                            # Store result
                            with self.buffer_lock:
                                self.processed_audio = result
                                self.ai_ready = True
                            
                            processing_time = time.time() - start_time
                            logger.info(f">>> AI processing completed in {processing_time:.1f}s")
                            
                        except Exception as e:
                            logger.error(f"AI processing error: {e}")
                            with self.buffer_lock:
                                self.processed_audio = None
                                self.ai_ready = False
                        
                        self.last_ai_time = current_time
                
                # Sleep to prevent busy waiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"AI worker error: {e}")
                time.sleep(1.0)
        
        logger.info("AI processing worker stopped")
    
    def apply_audio_transform(self, audio_data):
        """Apply the current transformation"""
        if self.processing_mode == "pass_through":
            return audio_data
        
        elif self.processing_mode == "volume_test":
            logger.debug("Applying volume reduction")
            return self.apply_volume_reduction(audio_data)
        
        elif self.processing_mode == "volume_boost":
            logger.debug("Applying volume boost")
            return self.apply_volume_boost(audio_data)
        
        elif self.processing_mode == "robot_voice":
            logger.debug("Applying robot effect")
            return self.apply_robot_effect(audio_data)
        
        elif self.processing_mode == "radio_effect":
            logger.debug("Applying radio effect")
            return self.apply_radio_effect(audio_data)
        
        elif self.processing_mode == "ai_transform":
            # For AI mode, use processed audio if available, otherwise pass through
            with self.buffer_lock:
                if self.ai_ready and self.processed_audio is not None:
                    # Use AI processed audio
                    result = self.processed_audio[:len(audio_data)] if len(self.processed_audio) >= len(audio_data) else audio_data
                    return result
                else:
                    # Pass through while AI is processing
                    return audio_data
        
        return audio_data
    
    def start_processing(self):
        """Start audio processing with NON-BLOCKING AI"""
        if self.is_processing:
            return
        
        if self.input_device is None or self.output_device is None:
            raise RuntimeError("Input and output devices must be selected first")
        
        self.is_processing = True
        self.audio_buffer = np.array([], dtype=np.float32)
        self.processed_audio = None
        self.ai_ready = False
        self.last_ai_time = time.time()
        self.ai_stop_event.clear()
        
        # Start AI processing thread if in AI mode
        if self.processing_mode == "ai_transform":
            self.ai_thread = threading.Thread(target=self.ai_separation_worker, daemon=True)
            self.ai_thread.start()
            logger.info("AI processing thread started")
        
        def audio_callback(indata, outdata, frames, time_info, status):
            if status:
                if status.input_overflow:
                    logger.warning("Input overflow")
                if status.output_underflow:
                    logger.warning("Output underflow")
            
            if not self.is_processing:
                outdata.fill(0)
                return
            
            try:
                # Get input audio (convert to mono)
                if indata.ndim > 1:
                    audio_input = indata[:, 0]
                else:
                    audio_input = indata.flatten()
                
                # Add to buffer for AI processing
                if self.processing_mode == "ai_transform":
                    with self.buffer_lock:
                        self.audio_buffer = np.concatenate([self.audio_buffer, audio_input])
                        # Keep buffer manageable
                        max_buffer = 5 * self.sample_rate  # 5 seconds max
                        if len(self.audio_buffer) > max_buffer:
                            self.audio_buffer = self.audio_buffer[-max_buffer:]
                
                # APPLY TRANSFORMATION
                processed_audio = self.apply_audio_transform(audio_input)
                
                # Output the transformed audio
                outdata[:, 0] = processed_audio[:frames]
                
                # Copy to other channels
                if outdata.shape[1] > 1:
                    for ch in range(1, outdata.shape[1]):
                        outdata[:, ch] = outdata[:, 0]
                
                # Log occasionally
                if np.random.random() < 0.03:  # 3% chance
                    input_vol = np.sqrt(np.mean(audio_input**2))
                    if input_vol > 0.001:
                        mode_info = f"{self.processing_mode.upper()}"
                        if self.processing_mode == "ai_transform":
                            mode_info += f" (AI Ready: {self.ai_ready})"
                        logger.info(f">>> {mode_info} - Audio level: {input_vol:.3f}")
                        
            except Exception as e:
                logger.error(f"Audio callback error: {e}")
                outdata.fill(0)
        
        try:
            # Get device info
            devices = sd.query_devices()
            input_name = devices[self.input_device]['name'] if self.input_device < len(devices) else f"Device {self.input_device}"
            output_name = devices[self.output_device]['name'] if self.output_device < len(devices) else f"Device {self.output_device}"
            
            logger.info(f"Starting audio processing - Mode: {self.processing_mode}")
            logger.info(f"Input: {input_name} (Device {self.input_device})")
            logger.info(f"Output: {output_name} (Device {self.output_device})")
            
            # Single stream for input and output
            self.output_stream = sd.Stream(
                callback=audio_callback,
                samplerate=self.sample_rate,
                blocksize=1024,
                device=(self.input_device, self.output_device),
                channels=1,
                dtype=np.float32,
                latency='low'
            )
            
            self.output_stream.start()
            
            logger.info(f">>> {self.processing_mode.upper()} MODE STARTED!")
            
            if self.processing_mode == "ai_transform":
                logger.info(">>> AI processing in background - audio may take a few seconds to transform")
            else:
                logger.info(">>> You should IMMEDIATELY hear the transformation!")
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            self.stop_processing()
            raise
    
    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        
        # Stop AI thread
        self.ai_stop_event.set()
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=2.0)
        
        # Stop audio stream
        if self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
        
        # Clear buffers
        self.audio_buffer = np.array([], dtype=np.float32)
        self.processed_audio = None
        self.ai_ready = False
        
        logger.info(">>> Processing stopped")
    
    def set_music_ratio(self, ratio):
        """Set separation ratio (for AI mode only)"""
        old_ratio = self.music_ratio
        self.music_ratio = max(0.0, min(1.0, ratio))
        
        # Reset AI state when ratio changes significantly
        if abs(old_ratio - self.music_ratio) > 0.2:
            with self.buffer_lock:
                self.ai_ready = False
                self.processed_audio = None
            logger.info(f"Music ratio changed to: {self.music_ratio:.1f} - AI reprocessing...")