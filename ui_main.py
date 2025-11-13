from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QSlider, QLabel, QPushButton, QComboBox, QGroupBox,
                            QProgressBar, QMessageBox, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPalette, QColor
import sounddevice as sd

class MainWindow(QMainWindow):
    music_ratio_changed = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.audio_processor = None
        self.init_ui()
        self.populate_audio_devices()
    
    def init_ui(self):
        self.setWindowTitle("Music Isolation Controller")
        self.setGeometry(300, 300, 400, 500)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("Music Isolation Controller")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Audio device selection
        device_group = QGroupBox("Audio Devices")
        device_layout = QVBoxLayout(device_group)
        
        # Input device
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Device:"))
        self.input_combo = QComboBox()
        input_layout.addWidget(self.input_combo)
        device_layout.addLayout(input_layout)
        
        # Output device
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Device:"))
        self.output_combo = QComboBox()
        output_layout.addWidget(self.output_combo)
        device_layout.addLayout(output_layout)
        
        layout.addWidget(device_group)
        
        # Music ratio control
        ratio_group = QGroupBox("Music/Vocal Ratio")
        ratio_layout = QVBoxLayout(ratio_group)
        
        # Labels for ratio extremes
        labels_layout = QHBoxLayout()
        labels_layout.addWidget(QLabel("VOCALS ONLY"))
        labels_layout.addStretch()
        labels_layout.addWidget(QLabel("BALANCED"))
        labels_layout.addStretch()
        labels_layout.addWidget(QLabel("MUSIC ONLY"))
        ratio_layout.addLayout(labels_layout)
        
        # Slider
        self.ratio_slider = QSlider(Qt.Horizontal)
        self.ratio_slider.setRange(0, 100)
        self.ratio_slider.setValue(50)  # Default to balanced
        self.ratio_slider.valueChanged.connect(self.on_ratio_changed)
        ratio_layout.addWidget(self.ratio_slider)
        
        # Current value display
        self.ratio_label = QLabel("Current: 50% Music")
        self.ratio_label.setAlignment(Qt.AlignCenter)
        ratio_font = QFont()
        ratio_font.setBold(True)
        self.ratio_label.setFont(ratio_font)
        ratio_layout.addWidget(self.ratio_label)
        
        layout.addWidget(ratio_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        
        self.stop_button = QPushButton("Stop Processing")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        
        layout.addLayout(button_layout)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout(status_group)
        
        self.status_label = QLabel("Ready to start")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        layout.addWidget(status_group)
        
        # Add some spacing at the bottom
        layout.addStretch()
    
    def populate_audio_devices(self):
        """Populate audio device dropdowns"""
        try:
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
            
            self.input_combo.clear()
            self.output_combo.clear()
            
            # Add default options
            self.input_combo.addItem("Default Input", None)
            self.output_combo.addItem("Default Output", None)
            
            # Add available devices
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.input_combo.addItem(f"{device['name']} ({host_apis[device['hostapi']]['name']})", i)
                if device['max_output_channels'] > 0:
                    self.output_combo.addItem(f"{device['name']} ({host_apis[device['hostapi']]['name']})", i)
                    
        except Exception as e:
            QMessageBox.warning(self, "Audio Device Error", f"Could not query audio devices: {e}")
    
    def on_ratio_changed(self, value):
        """Handle music ratio slider changes"""
        ratio = value / 100.0
        self.ratio_label.setText(f"Current: {value}% Music")
        
        if self.audio_processor:
            self.audio_processor.set_music_ratio(ratio)
    
    def start_processing(self):
        """Start audio processing"""
        try:
            from audio_processor import AudioProcessor
            
            # Get selected devices
            input_device = self.input_combo.currentData()
            output_device = self.output_combo.currentData()
            
            # Initialize audio processor
            self.audio_processor = AudioProcessor()
            
            # Start processing
            self.audio_processor.start_processing(input_device, output_device)
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Processing audio...")
            self.status_label.setStyleSheet("color: green;")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start audio processing: {e}")
    
    def stop_processing(self):
        """Stop audio processing"""
        if self.audio_processor:
            self.audio_processor.stop_processing()
            self.audio_processor = None
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopped")
        self.status_label.setStyleSheet("color: red;")

    def closeEvent(self, event):
        """Handle application close"""
        self.stop_processing()
        event.accept()