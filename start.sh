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

# Function to check if a process is running
check_process() {
    pgrep -f "$1" >/dev/null
    return $?
}

# Create required directories
mkdir -p logs
mkdir -p uploads
mkdir -p cache
mkdir -p models

# Install frontend dependencies if not already installed
if [ ! -d "frontend/node_modules" ]; then
    log_info "Installing frontend dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Download required model files if not exists
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
    log_info "Downloading facial landmarks predictor..."
    curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
fi

# Function to start the backend
start_backend() {
    log_info "Starting backend server..."
    if [ "$PRODUCTION" = "true" ]; then
        # Production mode with gunicorn
        python -m gunicorn --bind 0.0.0.0:${PORT:-3000} \
                 --workers 1 \
                 --threads 2 \
                 --timeout 120 \
                 --access-logfile logs/access.log \
                 --error-logfile logs/error.log \
                 --capture-output \
                 --enable-stdio-inheritance \
                 "backend.app:app" &
    else
        # Development mode
        python backend/app.py &
    fi
    BACKEND_PID=$!
    log_success "Backend started with PID: $BACKEND_PID"
}

# Function to start the frontend
start_frontend() {
    log_info "Starting frontend server..."
    cd frontend
    if [ "$PRODUCTION" = "true" ]; then
        # Production mode with serve
        npx serve -s dist -l ${FRONTEND_PORT:-5173} &
    else
        # Development mode
        npm run dev &
    fi
    FRONTEND_PID=$!
    cd ..
    log_success "Frontend started with PID: $FRONTEND_PID"
}

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if services are already running
if check_process "backend/app.py" || check_process "gunicorn"; then
    log_warning "Backend is already running"
else
    start_backend
fi

if check_process "npm run dev" || check_process "serve -s dist"; then
    log_warning "Frontend is already running"
else
    start_frontend
fi

# Function to cleanup processes on exit
cleanup() {
    log_info "Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Register cleanup function
trap cleanup SIGINT SIGTERM

log_success "Services started successfully!"
log_info "Backend running on http://localhost:${PORT:-3000}"
log_info "Frontend running on http://localhost:${FRONTEND_PORT:-5173}"
log_info "API documentation available at http://localhost:${PORT:-3000}/docs"
log_info "Press Ctrl+C to stop all services"

# Keep script running and monitor processes
while true; do
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        log_warning "Backend crashed, restarting..."
        start_backend
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        log_warning "Frontend crashed, restarting..."
        start_frontend
    fi
    sleep 5
done 