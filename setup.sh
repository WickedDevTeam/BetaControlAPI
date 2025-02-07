#!/bin/bash

# Set up colors for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored status messages
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_info "Setting up BetaControlAPI..."

# Create necessary directories
log_info "Creating required directories..."
mkdir -p logs
mkdir -p uploads
mkdir -p cache
mkdir -p models
mkdir -p frontend/dist

# Download and setup models
log_info "Setting up models..."
python backend/install_models.py

# Download required model files if not exists
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
    log_info "Downloading facial landmarks predictor..."
    curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
fi

# Set up frontend
log_info "Setting up frontend..."
cd frontend

# Install frontend dependencies if not already installed
if [ ! -d "node_modules" ]; then
    log_info "Installing frontend dependencies..."
    npm install
fi

# Build frontend for production
log_info "Building frontend for production..."
npm run build
cd ..

# Create default .env if it doesn't exist
if [ ! -f ".env" ]; then
    log_info "Creating default .env file..."
    cat > .env << EOL
FLASK_ENV=development
DEBUG=True
PRODUCTION=False
PORT=3000
HOST=0.0.0.0
MAX_CONTENT_LENGTH=16777216
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
MAX_CONCURRENT_USERS=10
RATE_LIMIT=100
ENABLE_CACHING=True
CACHE_TIMEOUT=3600
COMPRESSION_QUALITY=85
MAX_IMAGE_DIMENSION=4096
NUDENET_MODELS_PATH=/home/runner/workspace/models/nudenet
EOL
fi

# Make start script executable
chmod +x start.sh

log_success "Setup completed successfully!"
log_info "Run './start.sh' to start the application" 