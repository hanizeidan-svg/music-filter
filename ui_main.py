from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QSlider, QLabel, QPushButton, QComboBox, QGroupBox,
                            QMessageBox, QFrame, QRadioButton, QButtonGroup,
                            QTextEdit, QScrollArea, QGridLayout, QSplitter,
                            QProgressBar, QFileDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import sounddevice as sd
import logging
import os

from audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class ParameterSlider(QWidget):
    def __init__(self, param_name, initial_value=500, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        self.init_ui(initial_value)
    
    def init_ui(self, initial_value):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        
        self.label = QLabel(f"{self.param_name}: {initial_value}")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("font-weight: bold; font-size: 9pt;")
        layout.addWidget(self.label)
        
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(0, 1000)
        self.slider.setValue(initial_value)
        self.slider.setTickPosition(QSlider.TicksBothSides)
        self.slider.setTickInterval(100)
        self.slider.valueChanged.connect(self.on_slider_changed)
        layout.addWidget(self.slider)
    
    def on_slider_changed(self, value):
        self.label.setText(f"{self.param_name}: {value}")
        if hasattr(self, 'callback'):
            self.callback(self.param_name, value)
    
    def set_callback(self, callback):
        self.callback = callback
    
    def get_value(self):
        return self.slider.value()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_processor = None
        self.param_sliders = {}
        self.recording_timer = QTimer()
        self.recording_timer.timeout.connect(self.update_recording_status)
        self.init_ui()
        self.populate_audio_devices()
    
    def init_ui(self):
        self.setWindowTitle("Audio Processing Lab - With Recording")
        self.setGeometry(100, 100, 1200, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Title
        title = QLabel("🎛️ Audio Processing Lab")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin: 10px;")
        left_layout.addWidget(title)
        
        # Recording Controls
        recording_group = QGroupBox("🎤 Recording Controls")
        recording_layout = QVBoxLayout(recording_group)
        
        # Recording buttons
        record_btn_layout = QHBoxLayout()
        
        self.record_btn = QPushButton("🔴 START RECORDING")
        self.record_btn.clicked.connect(self.start_recording)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #c82333; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        
        self.stop_record_btn = QPushButton("⏹ STOP RECORDING")
        self.stop_record_btn.clicked.connect(self.stop_recording)
        self.stop_record_btn.setEnabled(False)
        self.stop_record_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #5a6268; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        
        record_btn_layout.addWidget(self.record_btn)
        record_btn_layout.addWidget(self.stop_record_btn)
        recording_layout.addLayout(record_btn_layout)
        
        # Recording status
        self.recording_status = QLabel("Not recording")
        self.recording_status.setAlignment(Qt.AlignCenter)
        self.recording_status.setStyleSheet("padding: 8px; background-color: #f8f9fa; border-radius: 3px; font-size: 9pt;")
        recording_layout.addWidget(self.recording_status)
        
        # Recording progress
        self.recording_progress = QProgressBar()
        self.recording_progress.setRange(0, 3600)  # Up to 1 hour
        self.recording_progress.setValue(0)
        self.recording_progress.setFormat("Recording: %v seconds")
        recording_layout.addWidget(self.recording_progress)
        
        left_layout.addWidget(recording_group)
        
        # Processing Mode
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self.mode_button_group = QButtonGroup(self)
        
        modes = [
            ("🔊 PASS-THROUGH", "pass_through", "Audio passes through unchanged"),
            ("🤖 AI TRANSFORM", "ai_transform", "AI vocal/music separation (uses parameter A)"),
            ("🔬 CUSTOM LAB", "custom_lab", "Custom function transformation")
        ]
        
        for text, mode, tooltip in modes:
            radio = QRadioButton(text)
            radio.setToolTip(tooltip)
            if mode == "custom_lab":
                radio.setChecked(True)
            self.mode_button_group.addButton(radio)
            mode_layout.addWidget(radio)
        
        left_layout.addWidget(mode_group)
        
        # Device Selection
        device_group = QGroupBox("Audio Devices")
        device_layout = QVBoxLayout(device_group)
        
        device_layout.addWidget(QLabel("INPUT Device:"))
        self.input_combo = QComboBox()
        self.input_combo.setToolTip("Audio source (e.g., CABLE Output)")
        device_layout.addWidget(self.input_combo)
        
        device_layout.addWidget(QLabel(""))  # Spacer
        
        device_layout.addWidget(QLabel("OUTPUT Device:"))
        self.output_combo = QComboBox()
        self.output_combo.setToolTip("Audio output (e.g., your speakers)")
        device_layout.addWidget(self.output_combo)
        
        cable_btn = QPushButton("Auto-Detect Virtual Cable")
        cable_btn.clicked.connect(self.auto_detect_virtual_cable)
        cable_btn.setStyleSheet("background-color: #17a2b8; color: white; padding: 5px;")
        device_layout.addWidget(cable_btn)
        
        left_layout.addWidget(device_group)
        
        # Control Buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("START PROCESSING")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                font-size: 12pt;
            }
            QPushButton:hover { background-color: #218838; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-weight: bold;
                padding: 12px;
                border-radius: 6px;
                font-size: 12pt;
            }
            QPushButton:hover { background-color: #c82333; }
            QPushButton:disabled { background-color: #6c757d; }
        """)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        left_layout.addLayout(button_layout)
        
        # Status
        self.status = QLabel("Ready to start - Select CUSTOM LAB mode for transformations")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("padding: 10px; background-color: #f8f9fa; border-radius: 5px; font-size: 10pt;")
        left_layout.addWidget(self.status)
        
        left_layout.addStretch()
        
        # Right panel - Custom Lab
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Custom Function Editor
        function_group = QGroupBox("Custom Function Editor")
        function_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        function_layout = QVBoxLayout(function_group)
        
        desc = QLabel("Use 'x' for audio data, parameters A-P (0-1000). Examples:")
        desc.setStyleSheet("color: #6c757d; font-size: 9pt;")
        function_layout.addWidget(desc)
        
        examples = QLabel(
            "x * A/500.0 | np.sin(x * B/100) | x * np.sin(x * C/10) | x + (x**3 * D/100000)"
        )
        examples.setStyleSheet("color: #17a2b8; font-size: 9pt; font-family: monospace; background-color: #f8f9fa; padding: 5px;")
        examples.setWordWrap(True)
        function_layout.addWidget(examples)
        
        self.function_edit = QTextEdit()
        self.function_edit.setPlainText("x * A/500.0")
        self.function_edit.setStyleSheet("font-family: 'Courier New'; font-size: 10pt;")
        self.function_edit.setMaximumHeight(100)
        function_layout.addWidget(self.function_edit)
        
        apply_btn = QPushButton("Apply Function")
        apply_btn.clicked.connect(self.apply_custom_function)
        apply_btn.setStyleSheet("background-color: #6f42c1; color: white; font-weight: bold; padding: 8px;")
        function_layout.addWidget(apply_btn)
        
        right_layout.addWidget(function_group)
        
        # Parameters Grid
        params_group = QGroupBox("Parameters A-P (0-1000)")
        params_layout = QGridLayout(params_group)
        
        for i in range(16):
            param_name = chr(65 + i)
            slider = ParameterSlider(param_name)
            slider.set_callback(self.on_parameter_changed)
            self.param_sliders[param_name] = slider
            params_layout.addWidget(slider, i // 4, i % 4)
        
        right_layout.addWidget(params_group)
        
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([400, 800])
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.addWidget(main_splitter)
        
        self.mode_button_group.buttonClicked.connect(self.on_mode_changed)
    
    def start_recording(self):
        """Start recording audio streams"""
        if not self.audio_processor or not self.audio_processor.is_processing:
            QMessageBox.warning(self, "Recording Error", "Please start audio processing first!")
            return
        
        try:
            self.audio_processor.start_recording()
            self.record_btn.setEnabled(False)
            self.stop_record_btn.setEnabled(True)
            self.recording_timer.start(1000)  # Update every second
            self.recording_status.setText("🔴 RECORDING - Capturing input and output streams")
            self.recording_status.setStyleSheet("color: #dc3545; background-color: #f8d7da; padding: 8px; border-radius: 3px; font-size: 9pt;")
            
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            QMessageBox.critical(self, "Recording Error", f"Failed to start recording: {e}")
    
    def stop_recording(self):
        """Stop recording and save files"""
        if not self.audio_processor:
            return
        
        try:
            # Stop recording
            input_data, output_data = self.audio_processor.stop_recording()
            
            # Stop timer
            self.recording_timer.stop()
            self.recording_progress.setValue(0)
            
            # Reset UI
            self.record_btn.setEnabled(True)
            self.stop_record_btn.setEnabled(False)
            self.recording_status.setText("Not recording")
            self.recording_status.setStyleSheet("padding: 8px; background-color: #f8f9fa; border-radius: 3px; font-size: 9pt;")
            
            # Ask for save location
            if len(input_data) > 0 or len(output_data) > 0:
                options = QFileDialog.Options()
                filename, _ = QFileDialog.getSaveFileName(
                    self, "Save Recording", "", "WAV Files (*.wav);;All Files (*)", options=options)
                
                if filename:
                    # Remove .wav extension if present
                    if filename.lower().endswith('.wav'):
                        filename = filename[:-4]
                    
                    # Save files
                    input_file, output_file = self.audio_processor.save_recording(
                        input_data, output_data, filename)
                    
                    if input_file and output_file:
                        QMessageBox.information(self, "Recording Saved", 
                                              f"Input stream saved as: {input_file}\n"
                                              f"Output stream saved as: {output_file}")
                    else:
                        QMessageBox.warning(self, "Save Error", "Failed to save recording files")
            else:
                QMessageBox.warning(self, "No Data", "No audio data was recorded")
                
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            QMessageBox.critical(self, "Recording Error", f"Failed to stop recording: {e}")
    
    def update_recording_status(self):
        """Update recording progress and status"""
        if self.audio_processor:
            status = self.audio_processor.get_recording_status()
            if status['is_recording']:
                seconds = int(status['duration'])
                self.recording_progress.setValue(seconds)
                
                input_sec = status['input_samples'] / self.audio_processor.sample_rate
                output_sec = status['output_samples'] / self.audio_processor.sample_rate
                
                self.recording_status.setText(
                    f"🔴 RECORDING - {seconds}s | "
                    f"Input: {input_sec:.1f}s | Output: {output_sec:.1f}s"
                )
    
    # ... (rest of the methods remain the same as previous version)
    
    def on_mode_changed(self):
        mode = self.get_processing_mode()
        if mode == "custom_lab":
            self.status.setText("CUSTOM LAB mode - Use function editor and parameters A-P")
        elif mode == "ai_transform":
            self.status.setText("AI TRANSFORM mode - Uses parameter A for vocal/music balance")
        else:
            self.status.setText("PASS-THROUGH mode - Audio unchanged")
    
    def on_parameter_changed(self, param_name, value):
        if self.audio_processor and self.audio_processor.is_processing:
            params = {param_name: value}
            self.audio_processor.set_custom_parameters(params)
    
    def apply_custom_function(self):
        function_code = self.function_edit.toPlainText()
        if self.audio_processor:
            self.audio_processor.set_custom_function(function_code)
            self.status.setText(f"Function applied: {function_code}")
    
    def populate_audio_devices(self):
        try:
            from audio_processor import AudioProcessor
            temp_processor = AudioProcessor()
            devices = temp_processor.get_audio_devices()
            
            self.input_combo.clear()
            self.output_combo.clear()
            
            for device in devices:
                device_text = f"{device['index']}: {device['name']}"
                
                if device['max_input_channels'] > 0:
                    self.input_combo.addItem(device_text, device['index'])
                
                if device['max_output_channels'] > 0:
                    self.output_combo.addItem(device_text, device['index'])
            
            self.auto_detect_virtual_cable()
            
        except Exception as e:
            self.status.setText(f"Error loading devices: {e}")
    
    def auto_detect_virtual_cable(self):
        try:
            for i in range(self.input_combo.count()):
                if any(cable in self.input_combo.itemText(i).lower() for cable in ['cable', 'vb-audio']):
                    self.input_combo.setCurrentIndex(i)
                    break
            
            for i in range(self.output_combo.count()):
                text = self.output_combo.itemText(i).lower()
                if not any(cable in text for cable in ['cable', 'vb-audio']):
                    if any(speaker in text for speaker in ['speaker', 'headphone', 'output']):
                        self.output_combo.setCurrentIndex(i)
                        break
            
            self.status.setText("Auto-detected devices. Adjust if needed.")
            
        except Exception as e:
            logger.error(f"Auto-detect error: {e}")
    
    def get_processing_mode(self):
        buttons = self.mode_button_group.buttons()
        for btn in buttons:
            if btn.isChecked():
                if "PASS-THROUGH" in btn.text():
                    return "pass_through"
                elif "AI TRANSFORM" in btn.text():
                    return "ai_transform"
                else:
                    return "custom_lab"
        return "custom_lab"
    
    def get_selected_devices(self):
        if self.input_combo.currentIndex() == -1 or self.output_combo.currentIndex() == -1:
            return None, None
        return self.input_combo.currentData(), self.output_combo.currentData()
    
    def start_processing(self):
        try:
            from audio_processor import AudioProcessor
            
            input_device, output_device = self.get_selected_devices()
            if input_device is None or output_device is None:
                self._handle_error("Please select both input and output devices")
                return
            
            processing_mode = self.get_processing_mode()
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status.setText(f"Starting {processing_mode} mode...")
            
            QTimer.singleShot(100, lambda: self._actually_start_processing(
                input_device, output_device, processing_mode))
            
        except Exception as e:
            self._handle_error(f"Startup failed: {e}")
    
    def _actually_start_processing(self, input_device, output_device, processing_mode):
        try:
            self.audio_processor = AudioProcessor()
            self.audio_processor.set_devices(input_device, output_device)
            self.audio_processor.set_processing_mode(processing_mode)
            
            initial_params = {name: slider.get_value() for name, slider in self.param_sliders.items()}
            self.audio_processor.set_custom_parameters(initial_params)
            
            if processing_mode == "custom_lab":
                self.audio_processor.set_custom_function(self.function_edit.toPlainText())
            
            self.audio_processor.start_processing()
            
            mode_display = processing_mode.replace('_', ' ').upper()
            self.status.setText(f"✅ {mode_display} ACTIVE! Processing audio...")
            self.status.setStyleSheet("color: green; background-color: #d4edda; padding: 10px; border-radius: 5px;")
            
        except Exception as e:
            self._handle_error(f"Processing failed: {e}")
    
    def _handle_error(self, error_msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText(f"❌ {error_msg}")
        self.status.setStyleSheet("color: red; background-color: #f8d7da; padding: 10px; border-radius: 5px;")
        
        QMessageBox.critical(self, "Error", error_msg)
    
    def stop_processing(self):
        try:
            if self.audio_processor:
                self.audio_processor.stop_processing()
                self.audio_processor = None
            
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status.setText("Stopped - Ready for new configuration")
            self.status.setStyleSheet("color: #6c757d; background-color: #f8f9fa; padding: 10px; border-radius: 5px;")
            
        except Exception as e:
            logger.error(f"Error stopping: {e}")

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()