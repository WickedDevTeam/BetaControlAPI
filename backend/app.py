from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from PIL import Image
import io
import logging
from logging.handlers import RotatingFileHandler
import os
import math
from typing import List, Tuple, Optional
import logging.config
import traceback
from datetime import datetime
import dlib
from setup import setup_all
from nudenet import NudeDetector
import random
from flask_restx import Api, Resource, fields, reqparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.install_models import setup_nudenet

# Run setup on import
try:
    setup_all()
except Exception as e:
    print(f"Warning: Automatic setup encountered some issues: {e}")
    print("The application will continue with reduced functionality.")

app = Flask(__name__)

# Configure Swagger documentation
api = Api(app, version='1.0', title='Image Censoring API',
          description='API for detecting and applying various censoring effects to sensitive areas in images',
          doc='/docs')

# Define Swagger models
detected_region = api.model('DetectedRegion', {
    'type': fields.String(required=True, description='Type of detected region'),
    'coords': fields.List(fields.Integer, required=True, description='Coordinates [x, y, width, height]'),
    'confidence': fields.Float(description='Detection confidence score'),
    'label': fields.String(description='Detailed label of the detected region'),
    'detection_type': fields.String(description='Type of detection method used')
})

process_response = api.model('ProcessImageResponse', {
    'processed_image': fields.String(required=True, description='Base64 encoded processed image data'),
    'regions': fields.List(fields.Nested(detected_region), required=True, description='List of detected regions')
})

process_request = api.model('ProcessImageRequest', {
    'image': fields.String(required=True, description='Base64 encoded image data with data URI scheme'),
    'effect': fields.String(enum=['pixelation', 'blur', 'blackbox', 'ruin', 'sticker'], 
                          default='pixelation', description='Type of censoring effect to apply'),
    'enabled_parts': fields.List(fields.String(enum=[
        'face', 'eyes', 'mouth', 'exposed_breast_f', 'covered_breast_f',
        'exposed_genitalia_f', 'covered_genitalia_f', 'exposed_breast_m',
        'exposed_genitalia_m', 'exposed_buttocks', 'covered_buttocks',
        'belly', 'feet'
    ]), description='List of body parts to detect and censor'),
    'strength': fields.Integer(min=1, max=10, default=7, description='Strength of the censoring effect (1-10)'),
    'sticker_category': fields.String(description='Category of stickers to use (only for sticker effect)')
})

error_response = api.model('ErrorResponse', {
    'error': fields.String(required=True, description='Error message describing what went wrong')
})

# Configure CORS properly
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": True
    }
})

# Configure detailed logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'color': {
            '()': 'colorlog.ColoredFormatter',
            'format': '%(log_color)s%(asctime)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s%(reset)s',
            'datefmt': '%Y-%m-%d %H:%M:%S',
            'log_colors': {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white'
            }
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'color',
            'level': 'DEBUG'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/app.log',
            'maxBytes': 1024 * 1024,  # 1MB
            'backupCount': 5,
            'formatter': 'detailed',
            'level': 'DEBUG'
        }
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

# Ensure logs directory exists
if not os.path.exists('logs'):
    os.makedirs('logs')

# Apply logging configuration
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

# Initialize Dlib's face detector and facial landmark predictor
face_detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    import urllib.request
    print("Downloading facial landmarks predictor...")
    urllib.request.urlretrieve(
        "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "shape_predictor_68_face_landmarks.dat.bz2"
    )
    import bz2
    with bz2.open("shape_predictor_68_face_landmarks.dat.bz2") as f:
        with open("shape_predictor_68_face_landmarks.dat", "wb") as out:
            out.write(f.read())
    os.remove("shape_predictor_68_face_landmarks.dat.bz2")

face_predictor = dlib.shape_predictor(predictor_path)

# Check for OpenPose installation
try:
    sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op
    # Initialize OpenPose
    params = {
        "model_folder": "models/",
        "face": True,
        "hand": True,
        "net_resolution": "-1x368"
    }
    
    # Check for OpenPose models
    models_dir = "models"
    required_models = [
        "pose/coco/pose_iter_440000.caffemodel",
        "face/pose_iter_116000.caffemodel",
        "hand/pose_iter_102000.caffemodel"
    ]
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print("\nOpenPose models not found. Please follow these steps to set up OpenPose:")
        print("\n1. Visit: https://github.com/CMU-Perceptual-Computing-Lab/openpose")
        print("2. Follow the installation instructions for your platform")
        print("3. Download the following model files and place them in the 'models' directory:")
        for model in required_models:
            print(f"   - {model}")
        print("\nFalling back to MediaPipe for pose detection until OpenPose is properly set up.")
        opWrapper = None
    else:
        missing_models = [model for model in required_models 
                         if not os.path.exists(os.path.join(models_dir, model))]
        if missing_models:
            print("\nSome OpenPose models are missing. Please download:")
            for model in missing_models:
                print(f"   - {model}")
            print("\nFalling back to MediaPipe for pose detection until all models are available.")
            opWrapper = None
        else:
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            print("OpenPose initialized successfully!")
except ImportError:
    print("\nOpenPose not found. To install OpenPose:")
    print("1. Clone the OpenPose repository:")
    print("   git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose")
    print("2. Follow the installation instructions for your platform")
    print("3. Make sure to build the Python API")
    print("\nFalling back to MediaPipe for pose detection.")
    opWrapper = None
except Exception as e:
    print(f"\nError initializing OpenPose: {e}")
    print("Falling back to MediaPipe for pose detection.")
    opWrapper = None

# Before initializing NudeDetector, ensure models are set up
setup_nudenet()

# Then initialize NudeDetector
nude_detector = NudeDetector()

def compute_convex_bounding_box(points, image_width, image_height, expansion_factor=0.1):
    """
    Given a list of (x,y) points, compute the convex hull and return an expanded bounding box.
    The expansion_factor expands the box by a fraction of its width and height.
    Returns a tuple: (x, y, w, h) clamped to the image boundaries.
    """
    pts = np.array(points, dtype=np.int32)
    if pts.size == 0:
        return 0, 0, 0, 0
    hull = cv2.convexHull(pts)
    x, y, w_box, h_box = cv2.boundingRect(hull)
    expand_w = int(w_box * expansion_factor)
    expand_h = int(h_box * expansion_factor)
    x = max(0, x - expand_w)
    y = max(0, y - expand_h)
    w_box = min(image_width - x, w_box + 2 * expand_w)
    h_box = min(image_height - y, h_box + 2 * expand_h)
    return x, y, w_box, h_box

def decode_image(image_data):
    """Decode base64 image data to OpenCV format"""
    logger.info("🔄 Starting image decoding process")
    try:
        encoded_data = image_data.split(',')[1] if ',' in image_data else image_data
        logger.debug("📝 Processing base64 encoded data")
        
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        logger.debug("🔍 Converting decoded data to numpy array")
        
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("❌ Failed to decode image: resulting image is None")
            raise ValueError("Failed to decode image")
            
        logger.info(f"✅ Successfully decoded image with shape: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"❌ Error decoding image: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise ValueError(f"Invalid image data: {str(e)}")

def encode_image(image):
    """Encode OpenCV image to base64"""
    logger.info("🔄 Starting image encoding process")
    try:
        logger.debug("📝 Encoding image to PNG format")
        _, buffer = cv2.imencode('.png', image)
        
        logger.debug("🔍 Converting to base64")
        encoded = base64.b64encode(buffer).decode('utf-8')
        
        logger.info("✅ Successfully encoded image")
        return 'data:image/png;base64,' + encoded
    except Exception as e:
        logger.error(f"❌ Error encoding image: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        raise ValueError(f"Failed to encode image: {str(e)}")

def ensure_bounds(x, y, w, h, img_width, img_height):
    """Ensure the region coordinates are within image bounds"""
    try:
        x = int(max(0, min(float(x), img_width - 1)))
        y = int(max(0, min(float(y), img_height - 1)))
        w = int(min(float(w), img_width - x))
        h = int(min(float(h), img_height - y))
        return x, y, w, h
    except Exception as e:
        logger.error(f"Error in ensure_bounds: {e}")
        logger.error(f"Inputs: x={x}, y={y}, w={w}, h={h}, img_width={img_width}, img_height={img_height}")
        raise

def apply_pixelation(image, region, strength):
    """Apply pixelation effect to a specific region"""
    try:
        x, y, w, h = ensure_bounds(*region, image.shape[1], image.shape[0])
        if w <= 0 or h <= 0:
            return image
            
        roi = image[y:y+h, x:x+w].copy()
        # Adjust block size based on strength (1-10)
        # Lower strength = larger blocks (more pixelation)
        # Decrease strength by 25% to make effect weaker
        reduced_strength = int(strength * 0.75)  # Reduce strength by 25%
        block_size = max(1, min(w, h) // max(1, (20 - reduced_strength * 2)))
        
        if block_size < 1:
            block_size = 1
            
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)))
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = pixelated
        return image
    except Exception as e:
        logger.error(f"Error in apply_pixelation: {e}")
        logger.error(f"Region: {region}, Strength: {strength}")
        return image

def apply_blur(image, region, strength):
    """Apply Gaussian blur to a specific region"""
    try:
        x, y, w, h = ensure_bounds(*region, image.shape[1], image.shape[0])
        if w <= 0 or h <= 0:
            return image
        
        roi = image[y:y+h, x:x+w].copy()
        # Adjust kernel size based on strength (1-10)
        # Higher strength = larger kernel (more blur)
        # Ensure kernel size is odd (required for Gaussian blur)
        # Double the strength for 100% stronger effect
        enhanced_strength = int(strength * 5.0)  # Increased from 3.125 to 5.0 for 100% stronger effect
        kernel_size = 2 * enhanced_strength + 1
        kernel_size = max(3, min(kernel_size, min(w, h) - 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Apply first blur pass
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        # Apply second blur pass for even stronger effect
        blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 0)
        
        image[y:y+h, x:x+w] = blurred
        return image
    except Exception as e:
        logger.error(f"Error in apply_blur: {e}")
        logger.error(f"Region: {region}, Strength: {strength}")
        return image

def apply_blackbox(image, region, strength=None):
    """Apply black box to a specific region"""
    try:
        x, y, w, h = ensure_bounds(*region, image.shape[1], image.shape[0])
        if w <= 0 or h <= 0:
            return image
        
        # Black box doesn't use strength parameter
        image[y:y+h, x:x+w] = [0, 0, 0]
        return image
    except Exception as e:
        logger.error(f"Error in apply_blackbox: {e}")
        logger.error(f"Region: {region}")
        return image

def get_randomized_ruin_parameters():
    """Get randomized parameters for the Ruin effect within ±20% of base values"""
    def randomize_value(base_value, variation=0.2):
        # Generate random factor between 0.8 and 1.2 (±20%)
        random_factor = 1.0 + (np.random.random() * 2 - 1) * variation
        return base_value * random_factor
    
    return {
        'pixelation_strength': randomize_value(5.5),
        'blur_strength': randomize_value(4.4),
        'rgb_shift_amount': randomize_value(12.5),
        'rgb_shift_red_angle': randomize_value(0.02),
        'rgb_shift_blue_angle': randomize_value(-0.02),
        'rgb_shift_green_angle': randomize_value(0.01),
        'rgb_shift_vertical': randomize_value(0.3),
        'noise_intensity': randomize_value(0.1875),
        'contrast': randomize_value(0.85),
        'brightness': randomize_value(-0.05),
        'global_blur_strength': randomize_value(0.05),
        'vignette_strength': randomize_value(0.15)
    }

def apply_rgb_shift(image, shift_amount=12.5, params=None):
    """Apply enhanced RGB channel shift effect to the image"""
    try:
        # Use default values if no params provided
        if params is None:
            params = {
                'rgb_shift_red_angle': 0.02,
                'rgb_shift_blue_angle': -0.02,
                'rgb_shift_green_angle': 0.01,
                'rgb_shift_vertical': 0.3
            }
        
        # Split the image into channels
        b, g, r = cv2.split(image)
        
        # Create shifted versions
        rows, cols = image.shape[:2]
        
        # Create transformation matrices with different shifts for each channel
        # Red channel: shift right and slightly down with randomized angles
        r_matrix = np.float32([
            [1, params['rgb_shift_red_angle'], shift_amount], 
            [0, 1, shift_amount * params['rgb_shift_vertical']]
        ])
        
        # Blue channel: shift left and slightly up with randomized angles
        b_matrix = np.float32([
            [1, params['rgb_shift_blue_angle'], -shift_amount], 
            [0, 1, -shift_amount * params['rgb_shift_vertical']]
        ])
        
        # Green channel: subtle diagonal shift with randomized angles
        g_matrix = np.float32([
            [1, params['rgb_shift_green_angle'], shift_amount * 0.15], 
            [0, 1, -shift_amount * params['rgb_shift_vertical'] * 0.5]
        ])
        
        # Apply the shifts with border replication to avoid black edges
        r = cv2.warpAffine(r, r_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        b = cv2.warpAffine(b, b_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        g = cv2.warpAffine(g, g_matrix, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        
        # Add subtle vignette effect to each channel
        center_x, center_y = cols/2, rows/2
        x = np.linspace(0, cols-1, cols)
        y = np.linspace(0, rows-1, rows)
        X, Y = np.meshgrid(x, y)
        
        # Calculate radial distance from center
        radius = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        max_radius = np.sqrt(center_x**2 + center_y**2)
        vignette = 1 - radius/max_radius * params.get('vignette_strength', 0.15)
        
        # Apply vignette to each channel slightly differently
        r = cv2.multiply(r.astype(float), vignette)
        g = cv2.multiply(g.astype(float), vignette * 1.05)  # Slightly stronger on green
        b = cv2.multiply(b.astype(float), vignette * 0.95)  # Slightly weaker on blue
        
        # Merge channels back with enhanced intensity
        merged = cv2.merge([
            np.clip(b, 0, 255).astype(np.uint8),
            np.clip(g, 0, 255).astype(np.uint8),
            np.clip(r, 0, 255).astype(np.uint8)
        ])
        
        return merged
    except Exception as e:
        logger.error(f"Error in apply_rgb_shift: {e}")
        logger.error(f"Parameters: {params}")
        return image

def apply_noise(image, intensity=0.1875):  # Increased from 0.15 to 0.1875 (25% more noise)
    """Add grain/noise effect to the image"""
    # Convert to float32
    img_float = image.astype(np.float32) / 255.0
    
    # Generate Gaussian noise
    noise = np.random.normal(0, intensity, image.shape)
    
    # Add noise to image
    noisy = img_float + noise
    
    # Clip values to valid range
    noisy = np.clip(noisy, 0, 1)
    
    # Convert back to uint8
    return (noisy * 255).astype(np.uint8)

def adjust_contrast_brightness(image, contrast=0.85, brightness=-0.05):  # Further reduced darkening from -0.075 to -0.05
    """Adjust image contrast and brightness"""
    # Convert to float32
    img_float = image.astype(np.float32) / 255.0
    
    # Apply contrast
    contrasted = img_float * contrast
    
    # Apply brightness
    brightened = contrasted + brightness
    
    # Clip values
    result = np.clip(brightened, 0, 1)
    
    # Convert back to uint8
    return (result * 255).astype(np.uint8)

def apply_ruin(image, region, strength=None, params=None):
    """Apply the Ruin effect to a specific region with optional randomized parameters"""
    try:
        if params is None:
            params = get_randomized_ruin_parameters()
            
        # Ensure coordinates are integers
        x, y, w, h = map(int, ensure_bounds(*region, image.shape[1], image.shape[0]))
        if w <= 0 or h <= 0:
            logger.warning(f"Invalid region dimensions: w={w}, h={h}")
            return image
            
        # Extract ROI
        roi = image[y:y+h, x:x+w].copy()
        
        try:
            # Calculate pixelation block size with randomized strength
            min_dimension = min(w, h)
            pixelation_strength = params['pixelation_strength'] * 0.5  # Reduced by 50%
            block_size = max(2, int(min_dimension // (20 - pixelation_strength * 2)))
            
            # Ensure block size is reasonable
            max_block_size = min_dimension // 4
            block_size = min(block_size, max_block_size)
            
            # Calculate new dimensions for pixelation
            small_w = max(2, w // block_size)
            small_h = max(2, h // block_size)
            
            # Apply pixelation
            small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Calculate blur kernel size with randomized strength
            blur_strength = params['blur_strength']
            kernel_size = int(2 * blur_strength + 1)
            kernel_size = max(3, min(kernel_size, min_dimension - 1))
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            # Apply double-pass blur
            blurred = cv2.GaussianBlur(pixelated, (kernel_size, kernel_size), 0)
            blurred = cv2.GaussianBlur(blurred, (kernel_size, kernel_size), 0)
            
            # Apply the processed ROI back to the image
            image[y:y+h, x:x+w] = blurred
            
        except Exception as effect_error:
            logger.error(f"Error applying effects: {effect_error}")
            logger.error(f"Region dimensions - w: {w}, h: {h}")
            logger.error(f"Block size: {block_size}, Kernel size: {kernel_size}")
            return image
        
        return image
    except Exception as e:
        logger.error(f"Error in apply_ruin: {e}")
        logger.error(f"Region: {region}")
        return image

def apply_global_blur(image, strength=0.05):
    """Apply a light blur effect to the entire image"""
    try:
        # Calculate kernel size based on image dimensions
        min_dimension = min(image.shape[0], image.shape[1])
        kernel_size = max(3, int(min_dimension * strength))
        
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Ensure kernel size is reasonable
        max_kernel = min(31, min_dimension // 10)  # Cap at 31 or 10% of min dimension
        kernel_size = min(kernel_size, max_kernel)
        
        # Apply blur
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    except Exception as e:
        logger.error(f"Error in apply_global_blur: {e}")
        logger.error(f"Image shape: {image.shape}, Calculated kernel size: {kernel_size}")
        return image

def process_ruin_effect(image, regions):
    """Process the entire image with the Ruin effect using randomized parameters"""
    try:
        # Generate randomized parameters once for consistent effect across the image
        params = get_randomized_ruin_parameters()
        logger.debug(f"Using randomized parameters: {params}")
        
        # First apply the regional effects
        for region in regions:
            try:
                # Handle both tuple and dict region formats
                if isinstance(region, tuple):
                    region_type, coords = region
                elif isinstance(region, dict):
                    coords = region.get('coords')
                else:
                    logger.warning(f"Invalid region format: {region}")
                    continue
                    
                if not coords or len(coords) != 4:
                    logger.warning(f"Invalid coordinates in region: {region}")
                    continue
                
                # Ensure coordinates are integers
                coords = tuple(map(int, coords))
                image = apply_ruin(image, coords, params=params)
                
            except Exception as region_error:
                logger.error(f"Error processing region in ruin effect: {region_error}")
                logger.error(f"Problematic region: {region}")
                continue
        
        # Then apply the global effects in sequence with randomized parameters
        try:
            # 1. RGB shift with randomized parameters
            image = apply_rgb_shift(image, shift_amount=params['rgb_shift_amount'], params=params)
        except Exception as e:
            logger.error(f"Error applying RGB shift: {e}")
            
        try:
            # 2. Noise/grain with randomized intensity
            image = apply_noise(image, intensity=params['noise_intensity'])
        except Exception as e:
            logger.error(f"Error applying noise: {e}")
            
        try:
            # 3. Contrast reduction and darkness adjustments with randomized values
            image = adjust_contrast_brightness(
                image, 
                contrast=params['contrast'],
                brightness=params['brightness']
            )
        except Exception as e:
            logger.error(f"Error adjusting contrast/brightness: {e}")
            
        try:
            # 4. Global blur effect with randomized strength
            image = apply_global_blur(image, strength=params['global_blur_strength'])
        except Exception as e:
            logger.error(f"Error applying global blur: {e}")
        
        return image
    except Exception as e:
        logger.error(f"Error in process_ruin_effect: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return image

def load_stickers():
    """Load stickers from the assets/Stickers directory"""
    stickers = {}
    sticker_dir = "assets/Stickers"
    
    try:
        # Get all subdirectories (categories)
        if not os.path.exists(sticker_dir):
            logger.error(f"Sticker directory not found: {sticker_dir}")
            return {}
            
        categories = [d for d in os.listdir(sticker_dir) 
                     if os.path.isdir(os.path.join(sticker_dir, d)) and not d.startswith('.')]
        
        if not categories:
            logger.warning("No sticker categories found")
            return {}
        
        for category in categories:
            category_path = os.path.join(sticker_dir, category)
            stickers[category] = []
            
            # Get all PNG files in the category
            png_files = [f for f in os.listdir(category_path) 
                        if f.lower().endswith('.png')]
            
            if not png_files:
                logger.warning(f"No PNG files found in category: {category}")
                continue
                
            for file in png_files:
                sticker_path = os.path.join(category_path, file)
                try:
                    # Load and validate sticker image
                    sticker_img = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
                    
                    if sticker_img is None:
                        logger.warning(f"Failed to load sticker: {file}")
                        continue
                        
                    # Validate image has alpha channel
                    if len(sticker_img.shape) != 3 or sticker_img.shape[2] != 4:
                        logger.warning(f"Skipping sticker without alpha channel: {file}")
                        continue
                        
                    # Validate image dimensions
                    if sticker_img.shape[0] > 1000 or sticker_img.shape[1] > 1000:
                        # Resize large stickers to prevent memory issues
                        scale = min(1000 / sticker_img.shape[0], 1000 / sticker_img.shape[1])
                        new_size = (int(sticker_img.shape[1] * scale), int(sticker_img.shape[0] * scale))
                        sticker_img = cv2.resize(sticker_img, new_size)
                        logger.info(f"Resized large sticker: {file}")
                    
                    stickers[category].append(sticker_img)
                    
                except Exception as e:
                    logger.error(f"Error loading sticker {file}: {e}")
                    continue
            
            # Remove categories with no valid stickers
            if not stickers[category]:
                del stickers[category]
        
        if not stickers:
            logger.error("No valid stickers loaded in any category")
            return {}
            
        logger.info(f"Successfully loaded stickers from categories: {list(stickers.keys())}")
        return stickers
    except Exception as e:
        logger.error(f"Error loading stickers: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return {}

# Load stickers on startup
STICKERS = load_stickers()

def apply_sticker(image, region, category=None, region_type=None):
    """Apply a random sticker from the selected category to the region, with underlying pixelation"""
    try:
        # Validate inputs
        if image is None or len(image.shape) != 3:
            logger.error("Invalid image format")
            return image
            
        x, y, w, h = ensure_bounds(*region, image.shape[1], image.shape[0])
        if w <= 0 or h <= 0:
            logger.warning("Invalid region dimensions")
            return image
            
        # First apply pixelation effect with random strength (3-5)
        pixelation_strength = random.randint(3, 5)
        image = apply_pixelation(image, region, pixelation_strength)
            
        # Validate category and get sticker
        if not STICKERS:
            logger.error("No stickers available")
            return image
            
        if not category or category not in STICKERS:
            available_categories = list(STICKERS.keys())
            if not available_categories:
                logger.error("No sticker categories available")
                return image
            category = available_categories[0]
            logger.warning(f"Using default category: {category}")
        
        if not STICKERS[category]:
            logger.warning(f"No stickers available in category {category}")
            return image
            
        sticker = random.choice(STICKERS[category])
        
        try:
            # Generate random rotation angle between -20 and 20 degrees
            angle = random.uniform(-20, 20)
            
            # Calculate dimensions to maintain aspect ratio while covering region
            sticker_h, sticker_w = sticker.shape[:2]
            
            # Add padding to prevent sticker from touching edges (10% padding)
            padding_x = int(w * 0.1)
            padding_y = int(h * 0.1)
            target_w = w - (2 * padding_x)
            target_h = h - (2 * padding_y)
            
            # Calculate scale to fit within padded area while maintaining aspect ratio
            scale_w = target_w / sticker_w
            scale_h = target_h / sticker_h
            # Use minimum scale to ensure sticker fits completely
            scale = min(scale_w, scale_h) * 0.9  # Additional 10% reduction for safety
            
            # Make stickers 100% larger for eyes and mouth
            if region_type in ['eyes', 'mouth']:
                scale *= 2.0  # Double the size
                logger.debug(f"Doubling sticker size for {region_type} region")
            
            # Limit maximum scale to prevent memory issues
            max_scale = 5.0
            if scale > max_scale:
                logger.warning(f"Limiting scale from {scale} to {max_scale}")
                scale = max_scale
            
            new_width = int(sticker_w * scale)
            new_height = int(sticker_h * scale)
            
            # Validate final dimensions
            if new_width <= 0 or new_height <= 0:
                logger.error("Invalid sticker dimensions after scaling")
                return image
                
            if new_width > image.shape[1] * 2 or new_height > image.shape[0] * 2:
                logger.error("Sticker dimensions too large after scaling")
                return image
            
            # Create rotation matrix
            center = (new_width // 2, new_height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Resize sticker
            resized_sticker = cv2.resize(sticker, (new_width, new_height))
            
            # Rotate sticker
            rotated_sticker = cv2.warpAffine(resized_sticker, rotation_matrix, (new_width, new_height))
            
            # Calculate placement to center the sticker over the region
            # Add padding to the placement calculation
            start_x = x + (w - new_width) // 2
            start_y = y + (h - new_height) // 2
            
            # Ensure the sticker stays within image bounds
            start_x = max(0, min(start_x, image.shape[1] - new_width))
            start_y = max(0, min(start_y, image.shape[0] - new_height))
            
            # Ensure rotated sticker has alpha channel
            if rotated_sticker.shape[2] != 4:
                logger.error("Rotated sticker lost alpha channel")
                return image
            
            # Create mask for alpha channel
            alpha = rotated_sticker[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            rgb = rotated_sticker[:, :, :3]
            
            # Calculate region in the image where sticker will be placed
            y1 = start_y
            y2 = min(start_y + new_height, image.shape[0])
            x1 = start_x
            x2 = min(start_x + new_width, image.shape[1])
            
            # Calculate corresponding region in the sticker
            sticker_y1 = 0
            sticker_y2 = y2 - y1
            sticker_x1 = 0
            sticker_x2 = x2 - x1
            
            try:
                # Blend sticker with image
                alpha_region = alpha[sticker_y1:sticker_y2, sticker_x1:sticker_x2]
                image_region = image[y1:y2, x1:x2]
                sticker_region = rgb[sticker_y1:sticker_y2, sticker_x1:sticker_x2]
                
                # Ensure all regions have the same dimensions
                if (alpha_region.shape[:2] != image_region.shape[:2] or 
                    alpha_region.shape[:2] != sticker_region.shape[:2]):
                    logger.error("Mismatched region dimensions")
                    return image
                
                image[y1:y2, x1:x2] = image_region * (1 - alpha_region) + sticker_region * alpha_region
                
            except Exception as e:
                logger.error(f"Error during image blending: {e}")
                return image
            
        except Exception as e:
            logger.error(f"Error during sticker transformation: {e}")
            return image
        
        return image
        
    except Exception as e:
        logger.error(f"Error applying sticker: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return image

EFFECTS = {
    'pixelation': apply_pixelation,
    'blur': apply_blur,
    'blackbox': apply_blackbox,
    'ruin': apply_ruin,
    'sticker': apply_sticker
}

def validate_region(x: int, y: int, size: int, w: int, h: int, min_size: int = 5) -> bool:
    """Validate if a region is valid and large enough to process"""
    # Check if region is within image bounds with some tolerance
    if x < -10 or y < -10 or x + size > w + 10 or y + size > h + 10:
        return False
        
    # Ensure region has some minimum absolute size
    if size < min_size:
        return False
        
    # Check if region dimensions are reasonable relative to image size
    region_area = size * size
    image_area = w * h
    min_area_ratio = 0.00005  # Allow very small regions (reduced from 0.0001)
    max_area_ratio = 0.95    # Allow very large regions (increased from 0.9)
    
    area_ratio = region_area / image_area
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False
        
    return True

def calculate_region_size(image_height: int, relative_size: float = 0.1, min_size: int = 20) -> int:
    """Calculate region size based on image dimensions with minimum size guarantee"""
    size = int(image_height * relative_size)
    return max(size, min_size)  # Ensure minimum size

def adjust_region_bounds(x: int, y: int, w: int, h: int, img_w: int, img_h: int, padding: int = 10) -> tuple:
    """Adjust region bounds to ensure they're within image bounds with padding"""
    x = max(-padding, min(x, img_w - w + padding))
    y = max(-padding, min(y, img_h - h + padding))
    w = min(w, img_w - x + padding)
    h = min(h, img_h - y + padding)
    return x, y, w, h

def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def smooth_coordinates(prev_coords: Optional[List[Tuple[int, int, int, int]]], 
                      curr_coords: List[Tuple[int, int, int, int]], 
                      smoothing_factor: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """Smooth coordinates between frames to reduce jitter"""
    if prev_coords is None or len(prev_coords) != len(curr_coords):
        return curr_coords
    
    smoothed = []
    for prev, curr in zip(prev_coords, curr_coords):
        smoothed_x = int(prev[0] * (1 - smoothing_factor) + curr[0] * smoothing_factor)
        smoothed_y = int(prev[1] * (1 - smoothing_factor) + curr[1] * smoothing_factor)
        smoothed_w = int(prev[2] * (1 - smoothing_factor) + curr[2] * smoothing_factor)
        smoothed_h = int(prev[3] * (1 - smoothing_factor) + curr[3] * smoothing_factor)
        smoothed.append((smoothed_x, smoothed_y, smoothed_w, smoothed_h))
    return smoothed

def detect_face_dlib(image, enabled_parts):
    """
    Detect facial features using Dlib for more accurate detection of specific features
    Returns list of regions as tuples: (type, (x, y, width, height))
    """
    regions = []
    h, w = image.shape[:2]
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_detector(gray)
        
        for face in faces:
            # Get facial landmarks
            landmarks = face_predictor(gray, face)
            points = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            # Use new detection area id 'face'
            if 'face' in enabled_parts:
                x, y = face.left(), face.top()
                width = face.width()
                height = face.height()
                
                # Expand the region more significantly for better face coverage
                expansion_ratio = 0.25  # 25% expansion
                expansion_w = int(width * expansion_ratio)
                expansion_h = int(height * expansion_ratio)
                
                # Adjust coordinates with expanded region
                x = max(0, x - expansion_w)
                y = max(0, y - expansion_h)  # Extra padding for forehead
                width = min(width + 2 * expansion_w, w - x)
                height = min(height + 2 * expansion_h, h - y)
                
                if validate_region(x, y, min(width, height), w, h):
                    regions.append(("face", (x, y, width, height)))
            
            # Only process eyes and mouth if face is not being processed
            if 'face' not in enabled_parts:
                # Eyes region
                if 'eyes' in enabled_parts:
                    # Left eye (points 36-41)
                    left_eye = points[36:42]
                    lx, ly, lw, lh = cv2.boundingRect(left_eye)
                    # Calculate eye center and dimensions
                    eye_width = lw * 3.4
                    eye_height = lh * 4.0
                    lcx = lx + lw/2
                    lcy = ly + lh/2
                    # Adjust coordinates to center the expanded box
                    lx = max(0, int(lcx - eye_width/2))
                    ly = max(0, int(lcy - eye_height/2))
                    lw = min(int(eye_width), w - lx)
                    lh = min(int(eye_height), h - ly)
                    if validate_region(lx, ly, min(lw, lh), w, h):
                        regions.append(("eyes", (lx, ly, lw, lh)))
                    
                    # Right eye (points 42-47)
                    right_eye = points[42:48]
                    rx, ry, rw, rh = cv2.boundingRect(right_eye)
                    # Calculate eye center and dimensions
                    eye_width = rw * 3.4
                    eye_height = rh * 4.0
                    rcx = rx + rw/2
                    rcy = ry + rh/2
                    # Adjust coordinates to center the expanded box
                    rx = max(0, int(rcx - eye_width/2))
                    ry = max(0, int(rcy - eye_height/2))
                    rw = min(int(eye_width), w - rx)
                    rh = min(int(eye_height), h - ry)
                    if validate_region(rx, ry, min(rw, rh), w, h):
                        regions.append(("eyes", (rx, ry, rw, rh)))
                
                # Mouth region
                if 'mouth' in enabled_parts:
                    # Mouth points (48-67)
                    mouth = points[48:68]
                    mx, my, mw, mh = cv2.boundingRect(mouth)
                    
                    # Calculate mouth center
                    mcx = mx + mw/2
                    mcy = my + mh/2
                    
                    # Expand mouth region
                    mouth_width = mw * 1.75
                    mouth_height = mh * 1.9
                    
                    # Adjust coordinates to center the expanded box
                    mx = max(0, int(mcx - mouth_width/2))
                    my = max(0, int(mcy - mouth_height/2))
                    mw = min(int(mouth_width), w - mx)
                    mh = min(int(mouth_height), h - my)
                    
                    if validate_region(mx, my, min(mw, mh), w, h):
                        regions.append(("mouth", (mx, my, mw, mh)))

        return regions
    except Exception as e:
        logger.error(f"Error in face detection: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return []

def expand_polygon(polygon: np.ndarray, expansion: float = 0.05) -> np.ndarray:
    """
    Expand (or contract if negative) the polygon radially by a certain fraction
    of its bounding box size. A more sophisticated method might use morphological
    dilation on a mask.
    """
    if polygon.shape[0] < 3:
        return polygon  # Can't expand a polygon with fewer than 3 points
    
    # Compute polygon center
    cx = np.mean(polygon[:, 0])
    cy = np.mean(polygon[:, 1])
    
    # Compute bounding box size
    min_x, min_y = np.min(polygon, axis=0)
    max_x, max_y = np.max(polygon, axis=0)
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y
    
    # Expand each vertex out from the center
    expanded = []
    for x, y in polygon:
        dx = x - cx
        dy = y - cy
        x_new = x + dx * expansion
        y_new = y + dy * expansion
        expanded.append([x_new, y_new])
    
    return np.array(expanded, dtype=np.int32)

def apply_effect_to_polygon(image, polygon_points, effect, strength):
    """
    Applies the chosen effect (pixelation, blur, blackbox, etc.) to the pixels
    within a polygon mask. This replaces bounding-box-only logic.
    """
    # Create a mask for the polygon
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_points], 255)
    
    # Extract ROI by bounding rectangle of polygon for efficiency
    x, y, w, h = cv2.boundingRect(polygon_points)
    roi = image[y:y+h, x:x+w].copy()
    roi_mask = mask[y:y+h, x:x+w]
    
    # Apply the chosen effect in that ROI for nonzero mask
    if effect == 'pixelation':
        # Increase block size for belly region to make the effect more pronounced
        block_size = max(1, min(w, h) // (20 - strength * 2))
        if 'belly' in str(polygon_points):  # Check if this is a belly region
            block_size = int(block_size * 1.35)  # Increase block size by 35%
        small = cv2.resize(roi, (w // block_size, h // block_size))
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        np.copyto(roi, pixelated, where=(roi_mask!=0)[...,None])

    elif effect == 'blur':
        # Increase kernel size for belly region
        kernel_size = 2 * strength + 1
        if 'belly' in str(polygon_points):  # Check if this is a belly region
            kernel_size = int(kernel_size * 1.35)  # Increase kernel size by 35%
        kernel_size = max(3, kernel_size)
        if kernel_size % 2 == 0:  # Ensure kernel size is odd
            kernel_size += 1
        blurred = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        np.copyto(roi, blurred, where=(roi_mask!=0)[...,None])

    elif effect == 'blackbox':
        roi[(roi_mask != 0)] = [0, 0, 0]
    
    # Put ROI back to original image
    image[y:y+h, x:x+w] = roi
    return image

def detect_body_openpose(image, enabled_parts):
    """
    Detect body parts using OpenPose for more accurate detection
    but now returning polygons instead of bounding boxes for some parts.
    """
    regions = []
    h, w = image.shape[:2]

    if opWrapper is None:
        return []

    datum = op.Datum()
    datum.cvInputData = image
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
        keypoints = datum.poseKeypoints[0]

        # Belly detection using torso keypoints
        if 'belly' in enabled_parts:
            # Get relevant keypoints for belly region
            neck = keypoints[1]
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[2]
            left_hip = keypoints[11]
            right_hip = keypoints[8]
            
            if all(pt[2] > 0.2 for pt in [neck, left_shoulder, right_shoulder, left_hip, right_hip]):
                # Create belly polygon using torso points
                belly_points = np.array([
                    [left_shoulder[0], left_shoulder[1]],
                    [right_shoulder[0], right_shoulder[1]],
                    [right_hip[0], right_hip[1]],
                    [left_hip[0], left_hip[1]],
                ], dtype=np.int32)
                
                # Expand the belly region by 35%
                belly_expanded = expand_polygon(belly_points, expansion=0.35)
                
                if belly_expanded.shape[0] >= 4:
                    # Convert polygon to bounding box for compatibility
                    x, y, w, h = cv2.boundingRect(belly_expanded)
                    # Add some padding
                    padding = int(min(w, h) * 0.1)
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)
                    
                    if validate_region(x, y, min(w, h), image.shape[1], image.shape[0]):
                        regions.append(('belly', (x, y, w, h)))

        # Genitalia detection
        if any('genitalia' in part for part in enabled_parts):
            left_hip = keypoints[11]
            right_hip = keypoints[8]
            left_knee = keypoints[12]
            right_knee = keypoints[9]

            if all(pt[2] > 0.2 for pt in [left_hip, right_hip, left_knee, right_knee]):
                # Create polygon for genital area
                genital_points = np.array([
                    [left_hip[0], left_hip[1]],
                    [left_knee[0], left_knee[1]*0.5 + left_hip[1]*0.5],
                    [right_knee[0], right_knee[1]*0.5 + right_hip[1]*0.5],
                    [right_hip[0], right_hip[1]],
                ], dtype=np.int32)

                # Expand the region by 35%
                genital_expanded = expand_polygon(genital_points, expansion=0.35)

                if genital_expanded.shape[0] >= 4:
                    x, y, w, h = cv2.boundingRect(genital_expanded)
                    padding = int(min(w, h) * 0.1)
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)
                    
                    if validate_region(x, y, min(w, h), image.shape[1], image.shape[0]):
                        regions.append(('genitalia', (x, y, w, h)))

    return regions

def detect_regions_nudenet(image, enabled_parts):
    """
    Detect sensitive regions using NudeNet's more accurate detection model (v3).
    Returns a list of region dictionaries with coordinates and confidence scores.
    """
    try:
        # Convert image to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3 and image.dtype == np.uint8:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get predictions from NudeNet
        detections = nude_detector.detect(image)
        logger.debug(f"NudeNet raw detections: {detections}")

        # Define confidence thresholds for different classes
        confidence_thresholds = {
            'FACE': 0.3,  # Lower threshold for faces since we have dlib as backup
            'BREAST': 0.2,  # Increased from 0.15 for more accurate breast detection
            'GENITALIA': 0.2,  # Increased from 0.15 for more accurate genitalia detection
            'BUTTOCKS': 0.2,  # Increased from 0.15 for more accurate buttocks detection
            'FEET': 0.25,  # Increased from 0.2 to reduce false positives
            'DEFAULT': 0.25  # Increased from 0.2 for better general accuracy
        }

        regions = []
        for detection in detections:
            try:
                # Validate detection format
                if not isinstance(detection, dict):
                    logger.warning(f"Skipping invalid detection format: {detection}")
                    continue
                if 'box' not in detection or 'score' not in detection or 'class' not in detection:
                    logger.warning(f"Detection missing required fields: {detection}")
                    continue

                box = detection['box']
                score = detection['score']
                class_name = detection['class'].upper()  # Ensure uppercase for comparison

                # Skip if this class isn't enabled
                if class_name not in enabled_parts:
                    logger.debug(f"Skipping {class_name} - not in enabled parts: {enabled_parts}")
                    continue

                # Get appropriate confidence threshold
                threshold = confidence_thresholds['DEFAULT']
                for key in confidence_thresholds:
                    if key in class_name:
                        threshold = confidence_thresholds[key]
                        break

                # Check confidence threshold
                if score <= threshold:
                    logger.debug(f"Skipping {class_name} - confidence {score} below threshold {threshold}")
                    continue

                # Convert box coordinates to integers and ensure proper format
                try:
                    x, y = int(box[0]), int(box[1])
                    w, h = int(box[2]), int(box[3])
                    
                    # Adjust region size based on class
                    if 'FACE' in class_name:
                        # Slightly expand face region
                        expansion = int(min(w, h) * 0.1)
                        x = max(0, x - expansion)
                        y = max(0, y - expansion)
                        w = min(w + 2 * expansion, image.shape[1] - x)
                        h = min(h + 2 * expansion, image.shape[0] - y)
                    elif 'BREAST' in class_name or 'GENITALIA' in class_name or 'BUTTOCKS' in class_name:
                        # Slightly expand sensitive areas for better coverage
                        expansion = int(min(w, h) * 0.15)
                        x = max(0, x - expansion)
                        y = max(0, y - expansion)
                        w = min(w + 2 * expansion, image.shape[1] - x)
                        h = min(h + 2 * expansion, image.shape[0] - y)
                    
                    # Create region with adjusted coordinates
                    region = {
                        'type': class_name.split('_')[0].lower(),  # Extract main type (face, breast, etc.)
                        'coords': (x, y, w, h),
                        'confidence': score,
                        'label': class_name,
                        'detection_type': 'nudenet'
                    }
                    
                    # Validate final coordinates
                    if w > 0 and h > 0 and x >= 0 and y >= 0:
                        regions.append(region)
                        logger.debug(f"Added {class_name} region with confidence {score} at {(x, y, w, h)}")
                    else:
                        logger.warning(f"Invalid coordinates for {class_name}: {(x, y, w, h)}")
                
                except (IndexError, ValueError) as e:
                    logger.error(f"Error processing box coordinates for {class_name}: {e}")
                    continue

            except Exception as detection_error:
                logger.error(f"Error processing individual detection: {detection_error}")
                logger.error(f"Problematic detection: {detection}")
                continue
        
        logger.debug(f"Processed {len(regions)} valid regions from NudeNet")
        return regions
    except Exception as e:
        logger.error(f"Error in NudeNet detection: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return []

def detect_regions(image, enabled_parts):
    """
    Enhanced detection function that combines NudeNet with existing detectors
    """
    regions = []
    
    try:
        # Normalize enabled_parts to match NudeNet's expectations
        normalized_parts = set()
        facial_parts = set()
        body_parts = set()
        
        for part in enabled_parts:
            # Facial features (handled by dlib)
            if part in ['face', 'eyes', 'mouth']:
                facial_parts.add(part)
                if part == 'face':
                    normalized_parts.add('FACE_FEMALE')
                    normalized_parts.add('FACE_MALE')
            
            # Female body parts (NudeNet)
            elif part == 'exposed_breast_f':
                normalized_parts.add('FEMALE_BREAST_EXPOSED')
            elif part == 'covered_breast_f':
                normalized_parts.add('FEMALE_BREAST_COVERED')
            elif part == 'exposed_genitalia_f':
                normalized_parts.add('FEMALE_GENITALIA_EXPOSED')
            elif part == 'covered_genitalia_f':
                normalized_parts.add('FEMALE_GENITALIA_COVERED')
            
            # Male body parts (NudeNet)
            elif part == 'exposed_breast_m':
                normalized_parts.add('MALE_BREAST_EXPOSED')
            elif part == 'exposed_genitalia_m':
                normalized_parts.add('MALE_GENITALIA_EXPOSED')
            
            # Buttocks (NudeNet)
            elif part == 'exposed_buttocks':
                normalized_parts.add('BUTTOCKS_EXPOSED')
            elif part == 'covered_buttocks':
                normalized_parts.add('BUTTOCKS_COVERED')
            
            # Additional body parts
            elif part == 'feet':
                normalized_parts.add('FEET_EXPOSED')
                normalized_parts.add('FEET_COVERED')
            elif part == 'belly':
                body_parts.add('belly')  # Handle belly with OpenPose

        logger.debug(f"Normalized parts for NudeNet: {normalized_parts}")
        logger.debug(f"Facial parts for dlib: {facial_parts}")
        logger.debug(f"Body parts for OpenPose: {body_parts}")
        
        # 1. First try NudeNet detection for body parts
        if normalized_parts:
            nudenet_regions = detect_regions_nudenet(image, normalized_parts)
            regions.extend(nudenet_regions)
            logger.debug(f"NudeNet detected {len(nudenet_regions)} regions")
        
        # 2. Use dlib for facial features if enabled
        if facial_parts:
            face_regions = detect_face_dlib(image, facial_parts)
            for face_region in face_regions:
                if isinstance(face_region, tuple):
                    region_type, coords = face_region
                    regions.append({
                        'type': region_type,
                        'coords': coords,
                        'confidence': 1.0,
                        'label': region_type,
                        'detection_type': 'dlib'
                    })
            logger.debug(f"Dlib detected {len(face_regions)} facial regions")
        
        # 3. Use OpenPose for belly and other body parts
        if body_parts:
            body_regions = detect_body_openpose(image, body_parts)
            for body_region in body_regions:
                if isinstance(body_region, tuple):
                    region_type, coords = body_region
                    regions.append({
                        'type': region_type,
                        'coords': coords,
                        'confidence': 1.0,
                        'label': region_type,
                        'detection_type': 'openpose'
                    })
            logger.debug(f"OpenPose detected {len(body_regions)} body regions")
        
        # 4. Validate and adjust all regions
        final_regions = []
        for region in regions:
            try:
                coords = region['coords']
                # Ensure coordinates are valid
                if len(coords) == 4:
                    x, y, w, h = coords
                    # Adjust bounds to image dimensions
                    x, y, w, h = adjust_region_bounds(x, y, w, h, image.shape[1], image.shape[0])
                    if w > 0 and h > 0:  # Only add if dimensions are valid
                        region['coords'] = (x, y, w, h)
                        final_regions.append(region)
                        logger.debug(f"Added validated region: {region['type']} at {(x, y, w, h)}")
            except Exception as e:
                logger.error(f"Error validating region {region}: {str(e)}")
                continue
        
        logger.debug(f"Final detection results: {len(final_regions)} regions")
        return final_regions
    except Exception as e:
        logger.error(f"Error in combined detection: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return []

def overlaps(region1, region2, threshold=0.5):
    """
    Check if two regions overlap significantly
    """
    x1, y1, w1, h1 = region1
    x2, y2, w2, h2 = region2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return False
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    
    # Return true if intersection over union exceeds threshold
    return intersection / min(area1, area2) > threshold

# Create namespaces
ns = api.namespace('', description='Image processing operations')

@ns.route('/process')
class ProcessImage(Resource):
    @ns.expect(process_request)
    @ns.response(200, 'Success', process_response)
    @ns.response(400, 'Validation Error', error_response)
    @ns.response(500, 'Server Error', error_response)
    def post(self):
        """Process image using combined detection system (NudeNet, dlib, and OpenPose)"""
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return {'error': 'No image data provided'}, 400

            image_data = data['image']
            effect = data.get('effect', 'pixelation')
            enabled_parts = data.get('enabled_parts', [])
            strength = data.get('strength', 7)
            sticker_category = data.get('sticker_category')

            logger.debug(f"Processing image with effect: {effect}, strength: {strength}")
            logger.debug(f"Enabled parts: {enabled_parts}")

            # If no detection toggles are selected, use default detection areas
            if not enabled_parts:
                if effect == 'ruin':
                    default_enabled = [
                        'face',
                        'exposed_breast_f', 'covered_breast_f',
                        'exposed_genitalia_f', 'covered_genitalia_f',
                        'covered_buttocks',
                        'feet'
                    ]
                else:
                    default_enabled = [
                        'face', 'eyes', 'mouth',
                        'exposed_breast_f', 'covered_breast_f',
                        'exposed_genitalia_f', 'covered_genitalia_f',
                        'exposed_buttocks', 'covered_buttocks',
                        'exposed_breast_m', 'exposed_genitalia_m',
                        'belly', 'feet'
                    ]
                enabled_parts = default_enabled
                logger.info("No detection toggles selected. Using default detection areas.")

            # Decode image
            img = decode_image(image_data)
            if img is None:
                return {'error': 'Failed to decode image'}, 400

            # Get regions using combined detection system
            regions = detect_regions(img, enabled_parts)
            logger.debug(f"Detected {len(regions)} regions")

            if not regions:
                logger.warning("No sensitive regions detected.")
            else:
                # Sort regions by size (largest first) to handle overlapping regions better
                def get_region_size(region):
                    if isinstance(region, tuple):
                        _, coords = region
                    else:
                        coords = region['coords']
                    return coords[2] * coords[3]  # width * height
                
                regions = sorted(regions, key=get_region_size, reverse=True)
                
                # Apply effects to detected regions
                if effect == 'ruin':
                    img = process_ruin_effect(img, regions)
                else:
                    for region in regions:
                        try:
                            logger.debug(f"Applying {effect} to region: {region}")
                            
                            # Extract coordinates and type based on region format
                            if isinstance(region, tuple):
                                region_type, coords = region
                            else:
                                region_type = region['type']
                                coords = region['coords']
                            
                            # Validate coordinates
                            if not coords or len(coords) != 4:
                                logger.warning(f"Invalid coordinates for region {region_type}: {coords}")
                                continue
                                
                            x, y, w, h = map(int, coords)
                            
                            # Skip invalid regions
                            if w <= 0 or h <= 0 or x < 0 or y < 0:
                                logger.warning(f"Invalid dimensions for region {region_type}: {coords}")
                                continue
                            
                            # Apply the effect with proper strength
                            effect_func = EFFECTS[effect]
                            if effect == 'sticker':
                                img = effect_func(img, (x, y, w, h), sticker_category, region_type)
                            else:
                                img = effect_func(img, (x, y, w, h), strength)
                            logger.debug(f"Applied {effect} to {region_type} at {coords}")
                            
                        except Exception as e:
                            logger.error(f"Error applying effect to region: {e}")
                            logger.error(f"Problematic region: {region}")
                            continue

            # Encode processed image
            processed_image_base64 = encode_image(img)
            if not processed_image_base64:
                return {'error': 'Failed to encode processed image'}, 500

            # Format regions for response
            formatted_regions = []
            for region in regions:
                try:
                    if isinstance(region, tuple):
                        formatted_regions.append({
                            'type': region[0],
                            'coords': region[1],
                            'confidence': 1.0,
                            'label': region[0],
                            'detection_type': 'default'
                        })
                    else:
                        formatted_regions.append({
                            'type': region['type'],
                            'coords': region['coords'],
                            'confidence': region.get('confidence', 1.0),
                            'label': region.get('label', region['type']),
                            'detection_type': region.get('detection_type', 'unknown')
                        })
                except Exception as e:
                    logger.error(f"Error formatting region: {e}")
                    continue

            return {
                'processed_image': processed_image_base64.split(',')[1],
                'regions': formatted_regions
            }

        except Exception as e:
            logger.error(f"Error in process_image: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return {'error': str(e)}, 500

@ns.route('/')
class HealthCheck(Resource):
    @ns.response(200, 'API is healthy')
    def get(self):
        """Check if the API is running and healthy"""
        logger.info("💓 Health check request received")
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }

if __name__ == '__main__':
    app.run(debug=True, port=8000) 