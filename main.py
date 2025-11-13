import sys
import os
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox
from ui_main import MainWindow

def exception_hook(exctype, value, traceback_obj):
    """Global exception handler"""
    error_msg = ''.join(traceback.format_exception(exctype, value, traceback_obj))
    print(f"Unhandled exception:\n{error_msg}")
    
    # Show error dialog
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("An unexpected error occurred")
    msg.setInformativeText(str(value))
    msg.setWindowTitle("Error")
    msg.setDetailedText(error_msg)
    msg.exec_()

def main():
    # Set exception hook
    sys.excepthook = exception_hook
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Music Isolation Controller")
    app.setApplicationVersion("1.0.0")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Start event loop
    try:
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()