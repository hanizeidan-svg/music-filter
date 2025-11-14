# from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
#                             QSlider, QLabel, QPushButton, QComboBox, QGroupBox,
#                             QMessageBox, QFrame, QProgressBar)
# from PyQt5.QtCore import Qt, pyqtSignal, QTimer
# from PyQt5.QtGui import QFont, QPalette, QColor
# import sounddevice as sd
# import logging

# logger = logging.getLogger(__name__)

# class MainWindow(QMainWindow):
#     music_ratio_changed = pyqtSignal(float)
    
#     def __init__(self):
#         super().__init__()
#         self.audio_processor = None
#         self.init_ui()
#         self.populate_audio_devices()
    
#     def init_ui(self):
#         self.setWindowTitle("Music Isolation Controller - AI Powered")
#         self.setGeometry(300, 300, 500, 600)
        
#         # Central widget
#         central_widget = QWidget()
#         self.setCentralWidget(central_widget)
        
#         # Main layout
#         layout = QVBoxLayout(central_widget)
#         layout.setSpacing(10)
#         layout.setContentsMargins(20, 20, 20, 20)
        
#         # Title
#         title = QLabel("AI Music Isolation Controller")
#         title.setAlignment(Qt.AlignCenter)
#         title_font = QFont()
#         title_font.setPointSize(18)
#         title_font.setBold(True)
#         title.setFont(title_font)
#         title.setStyleSheet("color: #2c3e50; margin-bottom: 10px;")
#         layout.addWidget(title)
        
#         subtitle = QLabel("Real-time Vocal/Music Separation using Demucs AI")
#         subtitle.setAlignment(Qt.AlignCenter)
#         subtitle_font = QFont()
#         subtitle_font.setPointSize(10)
#         subtitle.setFont(subtitle_font)
#         subtitle.setStyleSheet("color: #7f8c8d; margin-bottom: 20px;")
#         layout.addWidget(subtitle)
        
#         # Instructions
#         instructions_box = QGroupBox("🚀 How to Use")
#         instructions_layout = QVBoxLayout(instructions_box)
        
#         instructions_text = QLabel(
#             "1. Click 'Start AI Processing'\n"
#             "2. Play music from ANY app (Spotify, YouTube, etc.)\n" 
#             "3. Move slider: LEFT for vocals → RIGHT for music\n"
#             "4. The AI will separate vocals from music in REAL-TIME!"
#         )
#         instructions_text.setStyleSheet("color: #2c3e50; font-size: 10pt; padding: 10px;")
#         instructions_text.setAlignment(Qt.AlignLeft)
#         instructions_layout.addWidget(instructions_text)
        
#         layout.addWidget(instructions_box)
        
#         # Music ratio control
#         ratio_group = QGroupBox("🎚️ Vocal/Music Control")
#         ratio_group.setStyleSheet("QGroupBox { font-weight: bold; }")
#         ratio_layout = QVBoxLayout(ratio_group)
        
#         # Labels for ratio extremes
#         labels_layout = QHBoxLayout()
#         vocals_label = QLabel("🎤 PURE VOCALS")
#         vocals_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 10pt;")
#         labels_layout.addWidget(vocals_label)
        
#         labels_layout.addStretch()
        
#         balanced_label = QLabel("⚖️ BALANCED MIX")
#         balanced_label.setStyleSheet("color: #f39c12; font-weight: bold; font-size: 10pt;")
#         labels_layout.addWidget(balanced_label)
        
#         labels_layout.addStretch()
        
#         music_label = QLabel("🎵 PURE MUSIC")
#         music_label.setStyleSheet("color: #3498db; font-weight: bold; font-size: 10pt;")
#         labels_layout.addWidget(music_label)
        
#         ratio_layout.addLayout(labels_layout)
        
#         # Slider
#         self.ratio_slider = QSlider(Qt.Horizontal)
#         self.ratio_slider.setRange(0, 100)
#         self.ratio_slider.setValue(50)
#         self.ratio_slider.setTickPosition(QSlider.TicksBelow)
#         self.ratio_slider.setTickInterval(25)
#         self.ratio_slider.valueChanged.connect(self.on_ratio_changed)
#         self.ratio_slider.setStyleSheet("""
#             QSlider::groove:horizontal {
#                 border: 1px solid #999999;
#                 height: 12px;
#                 background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
#                     stop:0 #e74c3c, stop:0.5 #f39c12, stop:1 #3498db);
#                 margin: 2px 0;
#                 border-radius: 6px;
#             }
#             QSlider::handle:horizontal {
#                 background: #2c3e50;
#                 border: 2px solid #ffffff;
#                 width: 24px;
#                 margin: -8px 0;
#                 border-radius: 12px;
#             }
#         """)
#         ratio_layout.addWidget(self.ratio_slider)
        
#         # Current value display
#         self.ratio_label = QLabel("Current: 50% (Balanced Mix)")
#         self.ratio_label.setAlignment(Qt.AlignCenter)
#         ratio_font = QFont()
#         ratio_font.setBold(True)
#         ratio_font.setPointSize(12)
#         self.ratio_label.setFont(ratio_font)
#         self.ratio_label.setStyleSheet("color: #2c3e50; background-color: #ecf0f1; padding: 8px; border-radius: 5px;")
#         ratio_layout.addWidget(self.ratio_label)
        
#         layout.addWidget(ratio_group)
        
#         # Control buttons
#         button_layout = QHBoxLayout()
        
#         self.start_button = QPushButton("🤖 Start AI Processing")
#         self.start_button.clicked.connect(self.start_processing)
#         self.start_button.setStyleSheet("""
#             QPushButton {
#                 background-color: #27ae60;
#                 color: white;
#                 font-weight: bold;
#                 padding: 12px;
#                 border-radius: 6px;
#                 font-size: 12pt;
#             }
#             QPushButton:hover {
#                 background-color: #219a52;
#             }
#             QPushButton:disabled {
#                 background-color: #95a5a6;
#             }
#         """)
        
#         self.stop_button = QPushButton("⏹ Stop Processing")
#         self.stop_button.clicked.connect(self.stop_processing)
#         self.stop_button.setEnabled(False)
#         self.stop_button.setStyleSheet("""
#             QPushButton {
#                 background-color: #e74c3c;
#                 color: white;
#                 font-weight: bold;
#                 padding: 12px;
#                 border-radius: 6px;
#                 font-size: 12pt;
#             }
#             QPushButton:hover {
#                 background-color: #c0392b;
#             }
#             QPushButton:disabled {
#                 background-color: #95a5a6;
#             }
#         """)
        
#         button_layout.addWidget(self.start_button)
#         button_layout.addWidget(self.stop_button)
        
#         layout.addLayout(button_layout)
        
#         # Status
#         status_group = QGroupBox("📊 Status")
#         status_group.setStyleSheet("QGroupBox { font-weight: bold; }")
#         status_layout = QVBoxLayout(status_group)
        
#         self.status_label = QLabel("Ready to start AI audio processing")
#         self.status_label.setAlignment(Qt.AlignCenter)
#         self.status_label.setStyleSheet("padding: 12px; background-color: #f8f9fa; border-radius: 5px; font-size: 11pt;")
#         status_layout.addWidget(self.status_label)
        
#         layout.addWidget(status_group)
        
#         layout.addStretch()
    
#     def populate_audio_devices(self):
#         """Populate audio device dropdowns"""
#         try:
#             devices = sd.query_devices()
            
#             # We'll use automatic device detection, but still show available devices
#             input_count = sum(1 for dev in devices if dev['max_input_channels'] > 0)
#             output_count = sum(1 for dev in devices if dev['max_output_channels'] > 0)
            
#             logger.info(f"Found {input_count} input devices and {output_count} output devices")
            
#             self.status_label.setText(f"✅ Found {input_count} input and {output_count} output devices\nClick 'Start AI Processing' to begin!")
            
#         except Exception as e:
#             logger.error(f"Error with audio devices: {e}")
#             self.status_label.setText("⚠️ Audio device detection issue - trying automatic setup")
    
#     def on_ratio_changed(self, value):
#         """Handle music ratio slider changes"""
#         ratio = value / 100.0
        
#         # Update label with appropriate emoji and color
#         if value < 20:
#             display_text = f"🎤 {value}% (Vocal Mode)"
#             color = "#e74c3c"
#         elif value > 80:
#             display_text = f"🎵 {value}% (Music Mode)" 
#             color = "#3498db"
#         else:
#             display_text = f"⚖️ {value}% (Balanced)"
#             color = "#f39c12"
        
#         self.ratio_label.setText(display_text)
#         self.ratio_label.setStyleSheet(f"color: {color}; background-color: #ecf0f1; padding: 8px; border-radius: 5px;")
        
#         # Update audio processor
#         if self.audio_processor:
#             self.audio_processor.set_music_ratio(ratio)
    
#     def start_processing(self):
#         """Start audio processing"""
#         try:
#             from audio_processor import AudioProcessor
            
#             # Update UI immediately
#             self.start_button.setEnabled(False)
#             self.stop_button.setEnabled(True)
#             self.status_label.setText("🔄 Loading AI model...")
#             self.status_label.setStyleSheet("color: #f39c12; background-color: #fff3cd; padding: 12px; border-radius: 5px;")
            
#             # Force UI update
#             QTimer.singleShot(100, self._actually_start_processing)
            
#         except Exception as e:
#             error_msg = f"Failed to initialize: {e}"
#             logger.error(error_msg)
#             self._handle_start_error(error_msg)
    
#     def _actually_start_processing(self):
#         """Actually start processing after UI update"""
#         try:
#             # Initialize audio processor
#             self.audio_processor = AudioProcessor()
            
#             # Start processing
#             self.audio_processor.start_processing()
            
#             # Update UI
#             self.status_label.setText("✅ AI PROCESSING ACTIVE!\nPlay music and move the slider to hear separation!")
#             self.status_label.setStyleSheet("color: green; background-color: #d5f4e6; padding: 12px; border-radius: 5px; font-weight: bold;")
            
#         except Exception as e:
#             error_msg = f"Failed to start processing: {e}"
#             logger.error(error_msg)
#             self._handle_start_error(error_msg)
    
#     def _handle_start_error(self, error_msg):
#         """Handle start errors"""
#         self.start_button.setEnabled(True)
#         self.stop_button.setEnabled(False)
#         self.status_label.setText("❌ Failed to start processing")
#         self.status_label.setStyleSheet("color: red; background-color: #ffeaa7; padding: 12px; border-radius: 5px;")
        
#         # Show detailed error message
#         QMessageBox.critical(self, "Startup Error", 
#                            f"Could not start audio processing:\n\n{error_msg}\n\n"
#                            "Make sure your audio devices are working and try again.")
    
#     def stop_processing(self):
#         """Stop audio processing"""
#         try:
#             if self.audio_processor:
#                 self.audio_processor.stop_processing()
#                 self.audio_processor = None
            
#             # Update UI
#             self.start_button.setEnabled(True)
#             self.stop_button.setEnabled(False)
#             self.status_label.setText("⏹ Processing stopped - Ready to start again")
#             self.status_label.setStyleSheet("color: #7f8c8d; background-color: #f8f9fa; padding: 12px; border-radius: 5px;")
            
#             logger.info("Audio processing stopped by user")
            
#         except Exception as e:
#             logger.error(f"Error stopping processing: {e}")

#     def closeEvent(self, event):
#         """Handle application close"""
#         logger.info("Application closing...")
#         self.stop_processing()
#         event.accept()

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QSlider, QLabel, QPushButton, QComboBox, QGroupBox,
                            QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal
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
    
    def init_ui(self):
        self.setWindowTitle("AI Music Separator")
        self.setGeometry(300, 300, 400, 300)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("AI Music & Vocal Separator")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "1. Click START\n"
            "2. Play music from any app\n" 
            "3. Move slider to separate vocals/music"
        )
        instructions.setAlignment(Qt.AlignCenter)
        layout.addWidget(instructions)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self.on_slider_change)
        layout.addWidget(QLabel("🎤 Vocals ← SLIDER → Music 🎵"))
        layout.addWidget(self.slider)
        
        # Buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("START AI PROCESSING")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setStyleSheet("background-color: green; color: white; font-weight: bold;")
        
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        # Status
        self.status = QLabel("Ready to start")
        self.status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status)
    
    def on_slider_change(self, value):
        if self.audio_processor:
            self.audio_processor.set_music_ratio(value / 100.0)
    
    def start_processing(self):
        try:
            from audio_processor import AudioProcessor
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.status.setText("Loading AI model...")
            
            # Force UI update
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(100, self._actually_start)
            
        except Exception as e:
            self._handle_error(f"Startup failed: {e}")
    
    def _actually_start(self):
        try:
            self.audio_processor = AudioProcessor()
            self.audio_processor.start_processing()
            self.status.setText("✅ AI ACTIVE! Play music and move slider!")
            
        except Exception as e:
            self._handle_error(f"Processing failed: {e}")
    
    def _handle_error(self, error):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("❌ Failed - Check audio devices")
        QMessageBox.critical(self, "Error", error)
    
    def stop_processing(self):
        if self.audio_processor:
            self.audio_processor.stop_processing()
            self.audio_processor = None
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status.setText("Stopped - Ready to restart")
    
    def closeEvent(self, event):
        self.stop_processing()
        event.accept()