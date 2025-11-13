import torch
from demucs.pretrained import get_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model():
    try:
        logger.info("Attempting to download Demucs model...")
        model = get_model('htdemucs')
        logger.info("Model downloaded successfully!")
        
        # Check model details
        logger.info(f"Model type: {type(model)}")
        logger.info("Model is ready to use!")
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        logger.info("This might be due to:")
        logger.info("1. No internet connection")
        logger.info("2. Firewall blocking download")
        logger.info("3. Outdated demucs package")

if __name__ == "__main__":
    check_model()