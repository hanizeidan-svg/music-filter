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
        self.chunk_size = 1024  # Smaller chunks for real-time processing
        
        # Manual device selection
        self.input_device = None
        self.output_device = None
        
        # Processing mode
        self.processing_mode = "custom_lab"  # "pass_through", "ai_transform", or "custom_lab"
        
        # Audio buffers and synchronization
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_lock = threading.Lock()
        self.is_processing = False
        self.output_stream = None
        self.input_stream = None
        
        # Custom lab parameters
        self.custom_params = {chr(65 + i): 500 for i in range(16)}  # A-P: 0-1000, default 500
        self.custom_function = "x"  # Default: pass-through
        self.compiled_function = None
        self.last_function_hash = None
        
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
            # Normalize the function
            func_code = self.custom_function.strip()
            if not func_code:
                func_code = "x"
            
            # Create a safe environment for execution
            safe_dict = {
                'np': np,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'log10': np.log10,
                'sqrt': np.sqrt, 'abs': np.abs, 'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'pi': np.pi, 'e': np.e
            }
            
            # Add parameters to safe environment
            safe_dict.update(self.custom_params)
            
            # Create the function
            function_body = f"""
def transform_audio(x):
    return {func_code}
"""
            
            # Compile and execute
            exec_globals = {}
            exec(function_body, safe_dict, exec_globals)
            self.compiled_function = exec_globals['transform_audio']
            
            logger.info(f"Custom function compiled: {func_code}")
            
        except Exception as e:
            logger.error(f"Error compiling custom function: {e}")
            # Fallback to pass-through
            self.compiled_function = lambda x: x
    
    def apply_custom_transform(self, audio_data):
        """Apply custom transformation to audio data"""
        if self.compiled_function is None:
            return audio_data
        
        try:
            # Apply the custom function
            result = self.compiled_function(audio_data)
            
            # Ensure we get a numpy array back
            if not isinstance(result, np.ndarray):
                result = np.array(result, dtype=np.float32)
            
            # Ensure proper shape
            if result.shape != audio_data.shape:
                if len(result) == len(audio_data):
                    result = result.reshape(audio_data.shape)
                else:
                    logger.warning(f"Shape mismatch: input {audio_data.shape}, output {result.shape}")
                    return audio_data
            
            # Prevent extreme values
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result = result / max_val
                logger.debug("Normalized output to prevent clipping")
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying custom transform: {e}")
            return audio_data
    
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
            # Simple AI processing - you can enhance this
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
            
            # Use parameter A for mixing ratio (0-1000 maps to 0.0-1.0)
            ratio = self.custom_params.get('A', 500) / 1000.0
            result = (vocals_mono * (1 - ratio)) + (music_mono * ratio)
            
            # Ensure proper length
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
                    
                    # Keep buffer manageable
                    max_buffer_size = 2 * self.sample_rate
                    with self.buffer_lock:
                        if len(self.audio_buffer) > max_buffer_size:
                            self.audio_buffer = self.audio_buffer[-max_buffer_size:]
                    
                    # Log occasionally
                    if np.random.random() < 0.01:
                        volume = np.sqrt(np.mean(audio_input**2))
                        if volume > 0.01:
                            with self.buffer_lock:
                                buffer_size = len(self.audio_buffer)
                            logger.debug(f"Input: {volume:.3f} | Buffer: {buffer_size}")
                            
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
                        # Get audio chunk
                        audio_chunk = self.audio_buffer[:frames].copy()
                        self.audio_buffer = self.audio_buffer[frames:]
                    else:
                        # Not enough data
                        outdata.fill(0)
                        return
                
                # Apply processing based on mode
                if self.processing_mode == "pass_through":
                    processed_chunk = audio_chunk
                elif self.processing_mode == "ai_transform":
                    processed_chunk = self.apply_ai_separation(audio_chunk)
                else:  # custom_lab
                    processed_chunk = self.apply_custom_transform(audio_chunk)
                
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