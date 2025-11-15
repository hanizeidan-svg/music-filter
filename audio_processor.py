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
        self.custom_params = {chr(65 + i): 500 for i in range(16)}  # A-P: 0-1000, default 500
        self.custom_function = "x"  # Default: pass-through
        self.compiled_function = None
        
        # Predefined functions
        self.predefined_functions = {
            'multieffect': self.multieffect,
            'spectral': self.spectral,
            'vocal': self.vocal,
            'simplefx': self.simplefx
        }
        
        self.load_model()
    
    # ================= PREDEFINED FUNCTIONS =================
    
    def multieffect(self, x, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P):
        """Multi-Effect Processor: Comprehensive effects chain"""
        x = x * (A/500.0)  # Input gain
        
        # Bit crusher
        bit_depth = max(1, int(B / 31.25) + 1)
        quantization_levels = 2 ** bit_depth
        x = np.round(x * (quantization_levels/2)) / (quantization_levels/2)
        
        # Distortion
        distortion = C/1000.0
        x = np.tanh(x * (1 + distortion * 10)) / (1 + distortion)
        
        # Low-pass filter
        cutoff = 20 + (D/1000.0) * 19980
        nyquist = 44100/2
        normalized_cutoff = cutoff / nyquist
        filtered = np.zeros_like(x)
        for i in range(1, len(x)):
            filtered[i] = normalized_cutoff * x[i] + (1 - normalized_cutoff) * filtered[i-1]
        x = filtered
        
        # High-pass filter
        hpf_cutoff = 20 + (E/1000.0) * 4980
        hpf_normalized = hpf_cutoff / nyquist
        hpf_filtered = np.zeros_like(x)
        for i in range(1, len(x)):
            hpf_filtered[i] = 0.5 * (x[i] - x[i-1] + (1 - hpf_normalized) * hpf_filtered[i-1])
        x = hpf_filtered
        
        # Tremolo
        tremolo_rate = 0.1 + (F/1000.0) * 9.9
        tremolo_depth = 0.5
        t = np.arange(len(x)) / 44100.0
        tremolo = 1 - tremolo_depth * 0.5 * (1 + np.sin(2 * np.pi * tremolo_rate * t))
        x = x * tremolo
        
        # Chorus
        chorus_rate = 0.1 + (G/1000.0) * 4.9
        chorus_depth = 5
        chorus_delay = np.sin(2 * np.pi * chorus_rate * t) * chorus_depth
        chorus_out = np.zeros_like(x)
        for i in range(len(x)):
            delay_idx = int(i - chorus_delay[i])
            if delay_idx >= 0 and delay_idx < len(x):
                chorus_out[i] = 0.7 * x[i] + 0.3 * x[delay_idx]
        x = chorus_out
        
        # Continue with remaining effects...
        # [Include the rest of the multieffect implementation]
        
        # Output gain
        x = x * (P/500.0)
        return x
    
    def spectral(self, x, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P):
        """Spectral Manipulator: Frequency domain effects"""
        # Spectral shift
        spectral_shift = (A - 500) / 2.0
        
        # Spectral tilt
        tilt = (B - 500) / 500.0
        
        # Process in blocks (simplified)
        block_size = min(512, len(x))
        if block_size < 32:
            return x
            
        processed = np.zeros_like(x)
        
        for i in range(0, len(x) - block_size, block_size//2):
            block = x[i:i+block_size]
            window = np.hanning(len(block))
            windowed = block * window
            
            # Apply spectral processing
            # Low-mid boost
            low_mid_gain = D/500.0
            filtered_lm = np.zeros_like(windowed)
            for j in range(1, len(windowed)):
                filtered_lm[j] = 0.1 * windowed[j] + 0.9 * filtered_lm[j-1] * 0.5
            
            # Mid boost
            mid_gain = E/500.0
            filtered_mid = np.zeros_like(windowed)
            for j in range(2, len(windowed)):
                filtered_mid[j] = 0.3 * windowed[j] - 0.3 * windowed[j-2] + 1.4 * filtered_mid[j-1] - 0.7 * filtered_mid[j-2]
            
            # High boost
            high_gain = F/500.0
            filtered_high = np.zeros_like(windowed)
            for j in range(1, len(windowed)):
                filtered_high[j] = 0.5 * (windowed[j] - windowed[j-1] + 0.9 * filtered_high[j-1])
            
            combined = (filtered_lm * low_mid_gain + 
                       filtered_mid * mid_gain + 
                       filtered_high * high_gain)
            
            # Output saturation
            output_gain = P/500.0
            final = np.tanh(combined * output_gain)
            
            processed[i:i+block_size] += final * window
        
        return processed
    
    def vocal(self, x, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P):
        """Vocal Processor: Voice-specific effects"""
        x = x * (A/500.0)  # Input gain
        
        # De-esser
        deess_threshold = B/1000.0
        sibilance = np.zeros_like(x)
        for i in range(1, len(x)):
            sibilance[i] = 0.5 * (x[i] - x[i-1] + 0.9 * sibilance[i-1])
        x = np.where(np.abs(sibilance) > deess_threshold, x * 0.5, x)
        
        # Presence boost
        presence_gain = G/500.0
        presence_boost = np.zeros_like(x)
        for i in range(2, len(x)):
            presence_boost[i] = 0.5 * x[i] - 0.5 * x[i-2] + 1.8 * presence_boost[i-1] - 0.81 * presence_boost[i-2]
        x = x + presence_boost * presence_gain
        
        # Compression
        vocal_comp_threshold = K/1000.0
        vocal_comp_ratio = 3.0
        compressed_vocal = np.zeros_like(x)
        for i in range(len(x)):
            if abs(x[i]) > vocal_comp_threshold:
                compressed_vocal[i] = np.sign(x[i]) * (vocal_comp_threshold + (abs(x[i]) - vocal_comp_threshold) / vocal_comp_ratio)
            else:
                compressed_vocal[i] = x[i]
        x = compressed_vocal
        
        # Output gain
        output_gain = P/500.0
        x = np.tanh(x * output_gain)
        
        return x
    
    def simplefx(self, x, A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P):
        """SimpleFX: Minimalist multi-effect"""
        x = x * (A/500.0)  # Input gain
        
        # Drive
        drive = B/500.0
        x = np.tanh(x * (1 + drive))
        
        # Low-pass filter
        cutoff = 100 + (C/1000.0) * 19900
        alpha = cutoff / 44100.0
        filtered = np.zeros_like(x)
        for i in range(1, len(x)):
            filtered[i] = alpha * x[i] + (1 - alpha) * filtered[i-1]
        x = filtered
        
        # Tremolo
        tremolo_depth = E/1000.0
        tremolo_rate = 3
        t = np.arange(len(x)) / 44100.0
        tremolo = 1 - tremolo_depth * 0.5 * (1 + np.sin(2 * np.pi * tremolo_rate * t))
        x = x * tremolo
        
        # Delay
        delay_feedback = G/1000.0
        delay_time = 0.2
        delay_samples = int(44100 * delay_time)
        if delay_samples < len(x):
            for i in range(delay_samples, len(x)):
                x[i] += x[i - delay_samples] * delay_feedback
        
        # Output gain
        output_gain = P/500.0
        x = np.tanh(x * output_gain)
        
        return x
    
    # ================= PROCESSING METHODS =================
    
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
        """Compile the custom function with predefined functions available"""
        try:
            func_code = self.custom_function.strip()
            if not func_code:
                func_code = "x"
            
            # Create safe environment with predefined functions and parameters
            safe_dict = {
                'np': np,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'exp': np.exp, 'log': np.log, 'log10': np.log10,
                'sqrt': np.sqrt, 'abs': np.abs, 
                'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'pi': np.pi, 'e': np.e,
                
                # Predefined audio functions
                'multieffect': self.multieffect,
                'spectral': self.spectral, 
                'vocal': self.vocal,
                'simplefx': self.simplefx
            }
            
            # Add parameters
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
            # Apply the custom function with all parameters
            result = self.compiled_function(audio_data)
            
            # Ensure proper output
            if not isinstance(result, np.ndarray):
                result = np.array(result, dtype=np.float32)
            
            if result.shape != audio_data.shape:
                if len(result) == len(audio_data):
                    result = result.reshape(audio_data.shape)
                else:
                    logger.warning(f"Shape mismatch: input {audio_data.shape}, output {result.shape}")
                    return audio_data
            
            # Prevent clipping
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result = result / max_val
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying custom transform: {e}")
            return audio_data
    
    # [Rest of the class remains the same - get_audio_devices, set_devices, load_model, start_processing, etc.]
    
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
                    if indata.ndim > 1:
                        audio_input = indata[:, 0]
                    else:
                        audio_input = indata.flatten()
                    
                    with self.buffer_lock:
                        self.audio_buffer = np.concatenate([self.audio_buffer, audio_input])
                    
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
                
                outdata[:, 0] = processed_chunk[:frames]
                
                if outdata.shape[1] > 1:
                    for ch in range(1, outdata.shape[1]):
                        outdata[:, ch] = outdata[:, 0]
                        
            except Exception as e:
                logger.error(f"Output callback error: {e}")
                outdata.fill(0)
        
        try:
            devices = sd.query_devices()
            input_name = devices[self.input_device]['name'] if self.input_device < len(devices) else f"Device {self.input_device}"
            output_name = devices[self.output_device]['name'] if self.output_device < len(devices) else f"Device {self.output_device}"
            
            logger.info(f"Starting {self.processing_mode} mode")
            logger.info(f"Input: {input_name} (Device {self.input_device})")
            logger.info(f"Output: {output_name} (Device {self.output_device})")
            
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