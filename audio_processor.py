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
import soundfile as sf
from datetime import datetime

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TORCHAUDIO_USE_BACKEND_DISPATCHER'] = '1'

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self):
        self.model = None
        self.sample_rate = 44100
        self.chunk_size = 1024
        
        # Manual device selection
        self.input_device = None
        self.output_device = None
        
        # Processing mode
        self.processing_mode = "custom_lab"
        
        # Audio buffers and synchronization
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.is_processing = False
        self.output_stream = None
        self.input_stream = None
        
        # Custom lab parameters
        self.custom_params = {chr(65 + i): 500 for i in range(16)}
        self.custom_function = "x"
        self.compiled_function = None
        
        # Recording functionality
        self.is_recording = False
        self.recorded_input = []
        self.recorded_output = []
        self.recording_start_time = None
        self.recording_lock = threading.Lock()
        
        # AI model (optional)
        self.load_model()
    
    def set_processing_mode(self, mode):
        """Set processing mode: 'pass_through', 'ai_transform', or 'custom_lab'"""
        self.processing_mode = mode
        logger.info(f"Processing mode set to: {mode}")
    
    def set_custom_parameters(self, params_dict):
        """Set custom parameters A-P (0-1000)"""
        self.custom_params.update(params_dict)
        logger.info(f"Custom parameters updated: {params_dict}")
    
    def set_custom_function(self, function_code):
        """Set and compile custom function"""
        self.custom_function = function_code.strip()
        self.compile_custom_function()
    
    def compile_custom_function(self):
        """Compile the custom function for performance"""
        try:
            func_code = self.custom_function.strip()
            if not func_code:
                func_code = "x"
            
            safe_dict = {
                'np': np,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'log10': np.log10,
                'sqrt': np.sqrt, 'abs': np.abs, 'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'pi': np.pi, 'e': np.e
            }
            
            safe_dict.update(self.custom_params)
            
            function_body = f"""
def transform_audio(x):
    return {func_code}
"""
            
            exec_globals = {}
            exec(function_body, safe_dict, exec_globals)
            self.compiled_function = exec_globals['transform_audio']
            
            logger.info(f"Custom function compiled: {func_code}")
            
        except Exception as e:
            logger.error(f"Error compiling custom function: {e}")
            self.compiled_function = lambda x: x
    
    def apply_custom_transform(self, audio_data):
        """Apply custom transformation to audio data"""
        if self.compiled_function is None:
            return audio_data
        
        try:
            result = self.compiled_function(audio_data)
            
            if not isinstance(result, np.ndarray):
                result = np.array(result, dtype=np.float32)
            
            if result.shape != audio_data.shape:
                if len(result) == len(audio_data):
                    result = result.reshape(audio_data.shape)
                else:
                    logger.warning(f"Shape mismatch: input {audio_data.shape}, output {result.shape}")
                    return audio_data
            
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result = result / max_val
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying custom transform: {e}")
            return audio_data
    
    # Recording Methods
    def start_recording(self):
        """Start recording input and output streams"""
        with self.recording_lock:
            self.is_recording = True
            self.recorded_input = []
            self.recorded_output = []
            self.recording_start_time = time.time()
            logger.info(">>> Recording STARTED - capturing input and output streams")
    
    def stop_recording(self):
        """Stop recording and return recorded data"""
        with self.recording_lock:
            self.is_recording = False
            recording_duration = time.time() - self.recording_start_time
            logger.info(f">>> Recording STOPPED - duration: {recording_duration:.1f}s")
            
            # Convert to numpy arrays
            input_data = np.concatenate(self.recorded_input) if self.recorded_input else np.array([], dtype=np.float32)
            output_data = np.concatenate(self.recorded_output) if self.recorded_output else np.array([], dtype=np.float32)
            
            return input_data, output_data
    
    def save_recording(self, input_data, output_data, filename_prefix=None):
        """Save recorded data to WAV files"""
        try:
            if filename_prefix is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename_prefix = f"recording_{timestamp}"
            
            input_filename = f"{filename_prefix}_input.wav"
            output_filename = f"{filename_prefix}_output.wav"
            
            # Save input stream
            if len(input_data) > 0:
                sf.write(input_filename, input_data, self.sample_rate)
                logger.info(f"Saved input recording: {input_filename} ({len(input_data)} samples)")
            else:
                logger.warning("No input data to save")
            
            # Save output stream
            if len(output_data) > 0:
                sf.write(output_filename, output_data, self.sample_rate)
                logger.info(f"Saved output recording: {output_filename} ({len(output_data)} samples)")
            else:
                logger.warning("No output data to save")
            
            return input_filename, output_filename
            
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            return None, None
    
    def get_recording_status(self):
        """Get current recording status and statistics"""
        with self.recording_lock:
            if not self.is_recording:
                return {
                    'is_recording': False,
                    'duration': 0,
                    'input_samples': 0,
                    'output_samples': 0
                }
            
            duration = time.time() - self.recording_start_time
            input_samples = sum(len(chunk) for chunk in self.recorded_input)
            output_samples = sum(len(chunk) for chunk in self.recorded_output)
            
            return {
                'is_recording': True,
                'duration': duration,
                'input_samples': input_samples,
                'output_samples': output_samples
            }
    
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
        """Load Demucs model (optional)"""
        try:
            logger.info("Loading AI model...")
            self.model = get_model('htdemucs')
            self.model.eval()
            logger.info("AI model loaded successfully!")
        except Exception as e:
            logger.warning(f"AI model not available: {e}")
            self.model = None
    
    def ensure_stereo(self, audio_data):
        """Convert to stereo for Demucs"""
        if audio_data.ndim == 1:
            return np.column_stack((audio_data, audio_data))
        return audio_data
    
    def apply_ai_separation(self, audio_data):
        """Apply AI separation (for ai_transform mode)"""
        if self.model is None or len(audio_data) < 1000:
            return audio_data
        
        try:
            target_length = min(len(audio_data), 44100)
            process_data = audio_data[:target_length]
            
            stereo_audio = self.ensure_stereo(process_data)
            audio_tensor = torch.from_numpy(stereo_audio.T).float().unsqueeze(0)
            
            with torch.no_grad():
                sources = apply_model(self.model, audio_tensor, progress=False)
            
            vocals = sources[0, 3].numpy()
            music = (sources[0, 0] + sources[0, 1] + sources[0, 2]).numpy() / 3.0
            
            vocals_mono = np.mean(vocals.T, axis=1)
            music_mono = np.mean(music.T, axis=1)
            
            ratio = self.custom_params.get('A', 500) / 1000.0
            result = (vocals_mono * (1 - ratio)) + (music_mono * ratio)
            
            if len(result) < len(audio_data):
                result = np.pad(result, (0, len(audio_data) - len(result)))
            else:
                result = result[:len(audio_data)]
            
            return result
            
        except Exception as e:
            logger.error(f"AI processing error: {e}")
            return audio_data
    
    def start_processing(self):
        """Start audio processing"""
        if self.is_processing:
            return
        
        if self.input_device is None or self.output_device is None:
            raise RuntimeError("Input and output devices must be selected first")
        
        self.is_processing = True
        self.audio_buffer = np.array([], dtype=np.float32)
        
        def input_callback(indata, frames, time_info, status):
            if status:
                logger.debug(f"Input status: {status}")
            
            if self.is_processing:
                try:
                    # Get audio input (convert to mono)
                    if indata.ndim > 1:
                        audio_input = indata[:, 0]
                    else:
                        audio_input = indata.flatten()
                    
                    # Add to buffer
                    with self.buffer_lock:
                        self.audio_buffer = np.concatenate([self.audio_buffer, audio_input])
                    
                    # Record input stream if recording is active
                    with self.recording_lock:
                        if self.is_recording:
                            self.recorded_input.append(audio_input.copy())
                    
                    # Keep buffer manageable
                    max_buffer_size = 2 * self.sample_rate
                    with self.buffer_lock:
                        if len(self.audio_buffer) > max_buffer_size:
                            self.audio_buffer = self.audio_buffer[-max_buffer_size:]
                    
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
                    if len(self.audio_buffer) >= frames:
                        audio_chunk = self.audio_buffer[:frames].copy()
                        self.audio_buffer = self.audio_buffer[frames:]
                    else:
                        outdata.fill(0)
                        return
                
                # Apply processing based on mode
                if self.processing_mode == "pass_through":
                    processed_chunk = audio_chunk
                elif self.processing_mode == "ai_transform":
                    processed_chunk = self.apply_ai_separation(audio_chunk)
                else:  # custom_lab
                    processed_chunk = self.apply_custom_transform(audio_chunk)
                
                # Record output stream if recording is active
                with self.recording_lock:
                    if self.is_recording:
                        self.recorded_output.append(processed_chunk.copy())
                
                # Output processed audio
                outdata[:, 0] = processed_chunk[:frames]
                
                # Copy to other channels
                if outdata.shape[1] > 1:
                    for ch in range(1, outdata.shape[1]):
                        outdata[:, ch] = outdata[:, 0]
                        
            except Exception as e:
                logger.error(f"Output callback error: {e}")
                outdata.fill(0)
        
        try:
            # Get device info
            devices = sd.query_devices()
            input_name = devices[self.input_device]['name'] if self.input_device < len(devices) else f"Device {self.input_device}"
            output_name = devices[self.output_device]['name'] if self.output_device < len(devices) else f"Device {self.output_device}"
            
            logger.info(f"Starting {self.processing_mode} mode")
            logger.info(f"Input: {input_name} (Device {self.input_device})")
            logger.info(f"Output: {output_name} (Device {self.output_device})")
            
            # Start streams
            self.input_stream = sd.InputStream(
                callback=input_callback,
                samplerate=self.sample_rate,
                blocksize=1024,
                device=self.input_device,
                channels=1,
                dtype=np.float32
            )
            
            self.output_stream = sd.OutputStream(
                callback=output_callback,
                samplerate=self.sample_rate,
                blocksize=1024,
                device=self.output_device,
                channels=1,
                dtype=np.float32,
                latency='low'
            )
            
            self.input_stream.start()
            self.output_stream.start()
            
            logger.info(f">>> {self.processing_mode.upper()} MODE STARTED!")
            
        except Exception as e:
            logger.error(f"Error starting audio streams: {e}")
            self.stop_processing()
            raise
    
    def stop_processing(self):
        """Stop processing"""
        self.is_processing = False
        
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
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
        
        self.audio_buffer = np.array([], dtype=np.float32)
        logger.info(">>> Processing stopped")