from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QSlider, QLabel, QPushButton, QComboBox, QGroupBox,
                            QMessageBox, QFrame, QRadioButton, QButtonGroup)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
import sounddevice as sd
import logging

from audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_processor = None
        self.init_ui()
        self.populate_audio_devices()
    
    def init_ui(self):
        self.setWindowTitle("AI Music Separator - Device Selection")
        self.setGeometry(300, 300, 600, 550)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("AI Vocal/Music Separator")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Processing Mode Selection
        mode_group = QGroupBox("Processing Mode")
        mode_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        mode_layout = QVBoxLayout(mode_group)
        
        # Radio buttons for mode selection
        self.mode_button_group = QButtonGroup(self)
        
        self.pass_through_radio = QRadioButton("🔊 PASS-THROUGH MODE (Test Audio Routing)")
        self.pass_through_radio.setToolTip("Audio passes through unchanged - use to verify device configuration")
        self.pass_through_radio.setChecked(True)
        self.mode_button_group.addButton(self.pass_through_radio)
        
        self.transform_radio = QRadioButton("🤖 AI TRANSFORM MODE (Vocal/Music Separation)")
        self.transform_radio.setToolTip("AI processes audio to separate vocals from music")
        self.mode_button_group.addButton(self.transform_radio)
        
        mode_layout.addWidget(self.pass_through_radio)
        mode_layout.addWidget(self.transform_radio)
        
        # Mode description
        mode_desc = QLabel("Start with PASS-THROUGH to verify audio routing, then switch to AI TRANSFORM")
        mode_desc.setStyleSheet("color: #6c757d; font-size: 9pt; font-style: italic; padding: 5px;")
        mode_desc.setWordWrap(True)
        mode_layout.addWidget(mode_desc)
        
        layout.addWidget(mode_group)
        
        # Device Selection Group
        device_group = QGroupBox("Audio Device Selection")
        device_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        device_layout = QVBoxLayout(device_group)
        
        # Input Device Selection
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("INPUT Device (Audio Source):"))
        self.input_combo = QComboBox()
        self.input_combo.setToolTip("Select where audio comes FROM (e.g., CABLE Output, Stereo Mix)")
        input_layout.addWidget(self.input_combo)
        device_layout.addLayout(input_layout)
        
        device_layout.addWidget(QLabel(""))  # Spacer
        
        # Output Device Selection
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("OUTPUT Device (Speakers):"))
        self.output_combo = QComboBox()
        self.output_combo.setToolTip("Select where audio goes TO (e.g., your speakers/headphones)")
        output_layout.addWidget(self.output_combo)
        device_layout.addLayout(output_layout)
        
        # Auto-detect virtual cable button
        cable_button = QPushButton("Auto-Detect Virtual Cable")
        cable_button.clicked.connect(self.auto_detect_virtual_cable)
        cable_button.setStyleSheet("background-color: #17a2b8; color: white; padding: 5px;")
        device_layout.addWidget(cable_button)
        
        layout.addWidget(device_group)
        
        # Control Group (only visible in transform mode)
        self.control_group = QGroupBox("AI Separation Control")
        self.control_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        control_layout = QVBoxLayout(self.control_group)
        
        # Slider
        slider_labels = QHBoxLayout()
        slider_labels.addWidget(QLabel("🎤 VOCALS"))
        slider_labels.addStretch()
        slider_labels.addWidget(QLabel("⚖️ BALANCED"))
        slider_labels.addStretch()
        slider_labels.addWidget(QLabel("🎵 MUSIC"))
        control_layout.addLayout(slider_labels)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.on_slider_change)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
        """)
        control_layout.addWidget(self.slider)
        
        # Mode display
        self.mode_label = QLabel("Current: Balanced Mode")
        self.mode_label.setAlignment(Qt.AlignCenter)
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 14pt; color: #f39c12; padding: 5px;")
        control_layout.addWidget(self.mode_label)
        
        layout.addWidget(self.control_group)
        
        # Initially hide control group for pass-through mode
        self.control_group.setVisible(False)
        
        # Buttons
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
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        
        self.stop_btn = QPushButton("STOP PROCESSING")
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
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #6c757d;
            }
        """)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        # Status
        self.status = QLabel("Select processing mode and devices, then click START")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("padding: 12px; background-color: #f8f9fa; border-radius: 5px; font-size: 11pt;")
        layout.addWidget(self.status)
        
        # Instructions
        instructions = QLabel(
            "💡 WORKFLOW:\n"
            "1. Select PASS-THROUGH mode and verify audio routing works\n"
            "2. Switch to AI TRANSFORM mode for vocal/music separation\n"
            "3. For virtual cable: INPUT='CABLE Output', OUTPUT=your speakers"
        )
        instructions.setStyleSheet("color: #6c757d; font-size: 10pt; padding: 10px; background-color: #e9ecef; border-radius: 5px;")
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        layout.addStretch()
        
        # Connect mode change signal
        self.mode_button_group.buttonClicked.connect(self.on_mode_changed)
    
    def on_mode_changed(self):
        """Handle processing mode change"""
        if self.pass_through_radio.isChecked():
            self.control_group.setVisible(False)
            self.status.setText("PASS-THROUGH mode selected - audio will pass through unchanged")
        else:
            self.control_group.setVisible(True)
            self.status.setText("AI TRANSFORM mode selected - use slider to control vocal/music separation")
    
    def populate_audio_devices(self):
        """Populate audio device dropdowns"""
        try:
            from audio_processor import AudioProcessor
            temp_processor = AudioProcessor()
            devices = temp_processor.get_audio_devices()
            
            self.input_combo.clear()
            self.output_combo.clear()
            
            # Add devices to appropriate dropdowns
            for device in devices:
                device_text = f"{device['index']}: {device['name']} "
                
                if device['max_input_channels'] > 0:
                    input_text = device_text + f"({device['max_input_channels']} in)"
                    self.input_combo.addItem(input_text, device['index'])
                
                if device['max_output_channels'] > 0:
                    output_text = device_text + f"({device['max_output_channels']} out)"
                    self.output_combo.addItem(output_text, device['index'])
            
            # Try to set sensible defaults
            self.auto_detect_virtual_cable()
            
            logger.info(f"Loaded {self.input_combo.count()} input devices and {self.output_combo.count()} output devices")
            
        except Exception as e:
            logger.error(f"Error populating audio devices: {e}")
            self.status.setText(f"Error loading audio devices: {e}")
    
    def auto_detect_virtual_cable(self):
        """Try to auto-detect virtual cable devices"""
        try:
            # Look for virtual cable in input devices
            for i in range(self.input_combo.count()):
                item_text = self.input_combo.itemText(i).lower()
                if any(cable in item_text for cable in ['cable', 'vb-audio', 'virtual']):
                    self.input_combo.setCurrentIndex(i)
                    logger.info(f"Auto-selected input: {self.input_combo.itemText(i)}")
                    break
            
            # Look for speakers in output devices (not virtual cable)
            for i in range(self.output_combo.count()):
                item_text = self.output_combo.itemText(i).lower()
                if not any(cable in item_text for cable in ['cable', 'vb-audio', 'virtual']):
                    if any(speaker in item_text for speaker in ['speaker', 'headphone', 'output', 'primary']):
                        self.output_combo.setCurrentIndex(i)
                        logger.info(f"Auto-selected output: {self.output_combo.itemText(i)}")
                        break
            
            self.status.setText("Auto-detected virtual cable devices. Adjust if needed.")
            
        except Exception as e:
            logger.error(f"Auto-detect error: {e}")
    
    def on_slider_change(self, value):
        ratio = value / 100.0
        
        # Update mode label
        if value < 20:
            self.mode_label.setText("🎤 VOCAL MODE")
            self.mode_label.setStyleSheet("font-weight: bold; font-size: 14pt; color: #e74c3c; padding: 5px;")
        elif value > 80:
            self.mode_label.setText("🎵 MUSIC MODE")
            self.mode_label.setStyleSheet("font-weight: bold; font-size: 14pt; color: #3498db; padding: 5px;")
        else:
            self.mode_label.setText("⚖️ BALANCED MODE")
            self.mode_label.setStyleSheet("font-weight: bold; font-size: 14pt; color: #f39c12; padding: 5px;")
        
        # Update audio processor
        if self.audio_processor:
            self.audio_processor.set_music_ratio(ratio)
    
    def get_selected_devices(self):
        """Get currently selected input and output devices"""
        if self.input_combo.currentIndex() == -1 or self.output_combo.currentIndex() == -1:
            return None, None
        
        input_device = self.input_combo.currentData()
        output_device = self.output_combo.currentData()
        
        return input_device, output_device
    
    def get_processing_mode(self):
        """Get current processing mode"""
        if self.transform_radio.isChecked():
            return "transform"
        else:
            return "pass_through"
    
    def start_processing(self):
        try:
            from audio_processor import AudioProcessor
            
            # Get selected devices
            input_device, output_device = self.get_selected_devices()
            if input_device is None or output_device is None:
                self._handle_error("Please select both input and output devices")
                return
            
            # Get processing mode
            processing_mode = self.get_processing_mode()
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            
            mode_text = "PASS-THROUGH" if processing_mode == "pass_through" else "AI TRANSFORM"
            self.status.setText(f"Starting {mode_text} mode...")
            
            # Force UI update
            QTimer.singleShot(100, lambda: self._actually_start_processing(input_device, output_device, processing_mode))
            
        except Exception as e:
            self._handle_error(f"Startup failed: {e}")
    
    def _actually_start_processing(self, input_device, output_device, processing_mode):
        try:
            # Initialize audio processor
            self.audio_processor = AudioProcessor()
            
            # Set the selected devices
            self.audio_processor.set_devices(input_device, output_device)
            
            # Set processing mode
            self.audio_processor.set_processing_mode(processing_mode)
            
            # Start processing
            self.audio_processor.start_processing()
            
            # Update UI
            input_name = self.input_combo.currentText()
            output_name = self.output_combo.currentText()
            mode_text = "PASS-THROUGH" if processing_mode == "pass_through" else "AI TRANSFORM"
            
            if processing_mode == "pass_through":
                self.status.setText(f"✅ PASS-THROUGH ACTIVE!\nInput: {input_name}\nOutput: {output_name}\nAudio is passing through unchanged")
                self.status.setStyleSheet("color: #17a2b8; background-color: #d1ecf1; padding: 12px; border-radius: 5px; font-size: 11pt;")
            else:
                self.status.setText(f"✅ AI TRANSFORM ACTIVE!\nInput: {input_name}\nOutput: {output_name}\nMove slider for vocal/music separation!")
                self.status.setStyleSheet("color: green; background-color: #d4edda; padding: 12px; border-radius: 5px; font-size: 11pt;")
            
        except Exception as e:
            self._handle_error(f"Processing failed: {e}")
    
    def _handle_error(self, error_msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText(f"❌ {error_msg}")
        self.status.setStyleSheet("color: red; background-color: #f8d7da; padding: 12px; border-radius: 5px; font-size: 11pt;")
        
        QMessageBox.critical(self, "Audio Device Error", 
                           f"{error_msg}\n\n"
                           "Please check:\n"
                           "• Input device can capture audio\n"
                           "• Output device can play audio\n"
                           "• Devices support 44100 Hz sample rate\n"
                           "• No other app is using the devices")
    
    def stop_processing(self):
        try:
            if self.audio_processor:
                self.audio_processor.stop_processing()
                self.audio_processor = None
            
            # Update UI
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status.setText("Stopped - Ready to restart with new settings")
            self.status.setStyleSheet("color: #6c757d; background-color: #f8f9fa; padding: 12px; border-radius: 5px; font-size: 11pt;")
            
            logger.info("Processing stopped by user")
            
        except Exception as e:
            logger.error(f"Error stopping processing: {e}")

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()
