import os
import sys
import urllib.request
import logging
import json
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/model_installation.log')
    ]
)

# Constants
MODELS_DIR = Path('models/nudenet')
CLASSIFIER_MODEL_URL = "https://github.com/notAI-tech/NudeNet/releases/download/v0/classifier_model.onnx"
DETECTOR_MODEL_URL = "https://github.com/notAI-tech/NudeNet/releases/download/v0/detector_v2_default_checkpoint.onnx"

def download_file(url: str, dest_path: Path) -> None:
    """Download a file with progress indication"""
    try:
        logging.info(f"Downloading {url} to {dest_path}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\rDownloading: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()  # New line after progress
        logging.info(f"Successfully downloaded {dest_path.name}")
    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        raise

def setup_nudenet():
    """Set up NudeNet models and configuration"""
    try:
        # Create models directory
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download classifier model
        classifier_path = MODELS_DIR / "classifier_model.onnx"
        if not classifier_path.exists():
            download_file(CLASSIFIER_MODEL_URL, classifier_path)
        
        # Download detector model
        detector_path = MODELS_DIR / "detector_v2_default_checkpoint.onnx"
        if not detector_path.exists():
            download_file(DETECTOR_MODEL_URL, detector_path)
        
        # Create configuration file
        config = {
            "version": "2.0",
            "classifier_model": str(classifier_path.absolute()),
            "detector_model": str(detector_path.absolute()),
            "base_dir": str(MODELS_DIR.absolute())
        }
        
        config_path = MODELS_DIR / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Set environment variable
        os.environ['NUDENET_MODELS_PATH'] = str(MODELS_DIR.absolute())
        
        logging.info("NudeNet setup completed successfully")
        return True
    except Exception as e:
        logging.error(f"Error setting up NudeNet: {e}")
        return False

if __name__ == "__main__":
    if setup_nudenet():
        sys.exit(0)
    sys.exit(1) 