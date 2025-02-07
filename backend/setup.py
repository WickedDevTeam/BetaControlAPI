import os
import sys
import requests
import bz2
from tqdm import tqdm
import logging
import subprocess
import platform
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path to the backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
DLIB_MODEL_PATH = os.path.join(BACKEND_DIR, "shape_predictor_68_face_landmarks.dat")
DLIB_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
OPENPOSE_MODELS = {
    "pose/coco": {
        "url": "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
        "hash": "5f88d2f1a886cc12f03bb9e96c5e4691"
    },
    "pose/mpi": {
        "url": "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel",
        "hash": "2ca0990c7562bd7ae03f3f54674513f8"
    }
}

def download_file(url: str, dest_path: str, desc: str = None) -> None:
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=desc,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                pbar.update(size)
    except Exception as e:
        logger.error(f"Error downloading file from {url}: {str(e)}")
        # Clean up partial download if it exists
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise

def setup_dlib_model():
    """Download and setup Dlib facial landmark model"""
    if not os.path.exists(DLIB_MODEL_PATH):
        logger.info("Downloading Dlib facial landmarks model...")
        compressed_path = DLIB_MODEL_PATH + ".bz2"
        
        try:
            # Download compressed file
            download_file(DLIB_MODEL_URL, compressed_path, "Downloading Dlib model")
            
            # Decompress
            logger.info("Extracting Dlib model...")
            with bz2.open(compressed_path) as f_in, open(DLIB_MODEL_PATH, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            
            # Cleanup
            os.remove(compressed_path)
            logger.info("Dlib model setup complete!")
        except Exception as e:
            logger.error(f"Error setting up Dlib model: {str(e)}")
            if os.path.exists(compressed_path):
                os.remove(compressed_path)
            if os.path.exists(DLIB_MODEL_PATH):
                os.remove(DLIB_MODEL_PATH)
            raise
    else:
        logger.info("Dlib model already exists, skipping download.")

def verify_model_file(file_path: str, expected_hash: str = None) -> bool:
    """Verify if a model file exists and optionally check its hash"""
    if not os.path.exists(file_path):
        return False
    if expected_hash and compute_file_hash(file_path) != expected_hash:
        logger.warning(f"Hash mismatch for {file_path}")
        return False
    return True

def setup_models_directory():
    """Create and setup models directory structure"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        for subdir in ["pose/coco", "pose/mpi", "face", "hand"]:
            os.makedirs(os.path.join(MODELS_DIR, subdir), exist_ok=True)

def download_openpose_models():
    """Download OpenPose models"""
    setup_models_directory()
    
    for model_path, model_info in OPENPOSE_MODELS.items():
        dest_path = os.path.join(MODELS_DIR, model_path, os.path.basename(model_info["url"]))
        if not os.path.exists(dest_path):
            logger.info(f"Downloading OpenPose model: {model_path}")
            download_file(model_info["url"], dest_path, f"Downloading {model_path}")

def setup_all():
    """Run complete setup process"""
    logger.info("Starting automatic setup process...")
    success = True
    
    # Setup Dlib model
    try:
        setup_dlib_model()
    except Exception as e:
        logger.error(f"Error setting up Dlib model: {str(e)}")
        success = False
    
    if not success:
        logger.warning("Setup completed with some errors. The application may have reduced functionality.")
    else:
        logger.info("Setup completed successfully!")
    
    return success

if __name__ == "__main__":
    success = setup_all()
    sys.exit(0 if success else 1) 