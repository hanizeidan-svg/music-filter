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
        self.setWindowTitle("Audio Transformer - Test Modes")
        self.setGeometry(300, 300, 600, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        title = QLabel("Audio Transformer - Test Modes")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Processing Mode Selection
        mode_group = QGroupBox("🎛️ TRANSFORMATION MODES (TEST EACH ONE)")
        mode_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        mode_layout = QVBoxLayout(mode_group)
        
        # Radio buttons for mode selection
        self.mode_button_group = QButtonGroup(self)
        
        modes = [
            ("🔊 PASS-THROUGH", "pass_through", "Original audio unchanged - test routing"),
            ("🔉 VOLUME TEST", "volume_test", "50% quieter - VERY OBVIOUS"),
            ("🔊 VOLUME BOOST", "volume_boost", "200% louder - VERY OBVIOUS"), 
            ("🤖 ROBOT VOICE", "robot_voice", "Robot effect - VERY OBVIOUS"),
            ("📻 RADIO EFFECT", "radio_effect", "AM radio effect - VERY OBVIOUS"),
            ("🎵 AI SEPARATION", "ai_transform", "Vocal/Music separation")
        ]
        
        for text, mode, tooltip in modes:
            radio = QRadioButton(text)
            radio.setToolTip(tooltip)
            radio.mode = mode
            self.mode_button_group.addButton(radio)
            mode_layout.addWidget(radio)
        
        # Set default
        self.mode_button_group.buttons()[0].setChecked(True)
        
        layout.addWidget(mode_group)
        
        # Device Selection Group
        device_group = QGroupBox("🎧 AUDIO DEVICES")
        device_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        device_layout = QVBoxLayout(device_group)
        
        # Input Device Selection
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("INPUT (Audio Source):"))
        self.input_combo = QComboBox()
        self.input_combo.setToolTip("Where audio comes FROM (e.g., CABLE Output)")
        input_layout.addWidget(self.input_combo)
        device_layout.addLayout(input_layout)
        
        device_layout.addWidget(QLabel(""))  # Spacer
        
        # Output Device Selection
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("OUTPUT (Speakers):"))
        self.output_combo = QComboBox()
        self.output_combo.setToolTip("Where audio goes TO (your speakers/headphones)")
        output_layout.addWidget(self.output_combo)
        device_layout.addLayout(output_layout)
        
        # Auto-detect button
        cable_button = QPushButton("Auto-Detect Virtual Cable")
        cable_button.clicked.connect(self.auto_detect_virtual_cable)
        cable_button.setStyleSheet("background-color: #17a2b8; color: white; padding: 5px;")
        device_layout.addWidget(cable_button)
        
        layout.addWidget(device_group)
        
        # AI Control Group (only for AI mode)
        self.ai_control_group = QGroupBox("🎵 AI SEPARATION CONTROL")
        self.ai_control_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 12pt; }")
        ai_layout = QVBoxLayout(self.ai_control_group)
        
        # Slider
        slider_labels = QHBoxLayout()
        slider_labels.addWidget(QLabel("🎤 VOCALS"))
        slider_labels.addStretch()
        slider_labels.addWidget(QLabel("⚖️ BALANCED"))
        slider_labels.addStretch()
        slider_labels.addWidget(QLabel("🎵 MUSIC"))
        ai_layout.addLayout(slider_labels)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.on_slider_change)
        ai_layout.addWidget(self.slider)
        
        # Mode display
        self.mode_label = QLabel("Current: Balanced Mode")
        self.mode_label.setAlignment(Qt.AlignCenter)
        self.mode_label.setStyleSheet("font-weight: bold; font-size: 14pt; color: #f39c12; padding: 5px;")
        ai_layout.addWidget(self.mode_label)
        
        layout.addWidget(self.ai_control_group)
        self.ai_control_group.setVisible(False)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("🎵 START PROCESSING")
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
        
        self.stop_btn = QPushButton("⏹ STOP PROCESSING")
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
        self.status = QLabel("Select a transformation mode and devices, then click START")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setStyleSheet("padding: 12px; background-color: #f8f9fa; border-radius: 5px; font-size: 11pt;")
        layout.addWidget(self.status)
        
        # Instructions
        instructions = QLabel(
            "🎯 TEST PROCEDURE:\n"
            "1. Start with PASS-THROUGH - verify audio works\n"
            "2. Test VOLUME TEST - should be 50% quieter (OBVIOUS)\n" 
            "3. Test ROBOT VOICE - should sound robotic (OBVIOUS)\n"
            "4. Test AI SEPARATION - move slider for vocal/music control"
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
        selected_button = self.mode_button_group.checkedButton()
        mode = selected_button.mode
        
        # Show/hide AI controls
        if mode == "ai_transform":
            self.ai_control_group.setVisible(True)
            self.status.setText("AI SEPARATION mode - use slider to control vocal/music balance")
        else:
            self.ai_control_group.setVisible(False)
            mode_name = selected_button.text().split(' ')[1]  # Get mode name from button text
            self.status.setText(f"{mode_name} mode - you should hear obvious audio changes")
    
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
                device_text = f"{device['index']}: {device['name']}"
                
                if device['max_input_channels'] > 0:
                    self.input_combo.addItem(device_text, device['index'])
                
                if device['max_output_channels'] > 0:
                    self.output_combo.addItem(device_text, device['index'])
            
            # Try to set sensible defaults
            self.auto_detect_virtual_cable()
            
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
                    break
            
            # Look for speakers in output devices
            for i in range(self.output_combo.count()):
                item_text = self.output_combo.itemText(i).lower()
                if not any(cable in item_text for cable in ['cable', 'vb-audio', 'virtual']):
                    if any(speaker in item_text for speaker in ['speaker', 'headphone', 'output']):
                        self.output_combo.setCurrentIndex(i)
                        break
            
            self.status.setText("Auto-detected devices. Adjust if needed.")
            
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
        selected_button = self.mode_button_group.checkedButton()
        return selected_button.mode
    
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
            
            mode_name = self.mode_button_group.checkedButton().text().split(' ')[1]
            self.status.setText(f"Starting {mode_name} mode...")
            
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
            mode_name = self.mode_button_group.checkedButton().text().split(' ')[1]
            self.status.setText(f"✅ {mode_name} MODE ACTIVE!\nYou should hear obvious audio changes!")
            
            if processing_mode == "pass_through":
                self.status.setStyleSheet("color: #17a2b8; background-color: #d1ecf1; padding: 12px; border-radius: 5px; font-size: 11pt;")
            else:
                self.status.setStyleSheet("color: green; background-color: #d4edda; padding: 12px; border-radius: 5px; font-size: 11pt;")
            
        except Exception as e:
            self._handle_error(f"Processing failed: {e}")
    
    def _handle_error(self, error_msg):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText(f"❌ {error_msg}")
        self.status.setStyleSheet("color: red; background-color: #f8d7da; padding: 12px; border-radius: 5px; font-size: 11pt;")
        
        QMessageBox.critical(self, "Audio Error", 
                           f"{error_msg}\n\n"
                           "Check device selection and try again.")
    
    def stop_processing(self):
        try:
            if self.audio_processor:
                self.audio_processor.stop_processing()
                self.audio_processor = None
            
            # Update UI
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status.setText("Stopped - Ready to test another mode")
            self.status.setStyleSheet("color: #6c757d; background-color: #f8f9fa; padding: 12px; border-radius: 5px; font-size: 11pt;")
            
        except Exception as e:
            logger.error(f"Error stopping processing: {e}")

    def closeEvent(self, event):
        self.stop_processing()
        event.accept()