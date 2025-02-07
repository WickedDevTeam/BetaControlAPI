import os
import urllib.request
import bz2
import logging
import sys

def setup_all():
    """Set up all required components for the application."""
    setup_logging()
    setup_directories()
    download_models()

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/setup.log')
        ]
    )

def setup_directories():
    """Create necessary directories."""
    directories = ['logs', 'uploads', 'cache', 'models']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def download_models():
    """Download required model files."""
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        logging.info("Downloading facial landmarks predictor...")
        try:
            urllib.request.urlretrieve(
                "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
                "shape_predictor_68_face_landmarks.dat.bz2"
            )
            with bz2.open("shape_predictor_68_face_landmarks.dat.bz2") as f:
                with open("shape_predictor_68_face_landmarks.dat", "wb") as out:
                    out.write(f.read())
            os.remove("shape_predictor_68_face_landmarks.dat.bz2")
            logging.info("Successfully downloaded and extracted facial landmarks predictor")
        except Exception as e:
            logging.error(f"Error downloading facial landmarks predictor: {e}")
            raise

if __name__ == "__main__":
    setup_all() 