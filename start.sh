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

# Function to stop servers on script termination
cleanup() {
    log_info "Stopping servers..."
    pkill -f "python backend/app.py" || true
    pkill -f "npm run dev" || true
    exit
}

# Function to check if a port is in use
is_port_in_use() {
    lsof -i :$1 >/dev/null 2>&1
    return $?
}

# Function to wait for a service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=$3
    local attempt=1

    log_info "Waiting for $service_name to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            log_success "$service_name is ready!"
            return 0
        fi
        
        # Check if the process is still running
        if ! pgrep -f "$4" >/dev/null; then
            log_error "$service_name process has died"
            return 1
        fi
        
        log_info "Attempt $attempt/$max_attempts: $service_name not ready yet, waiting..."
        sleep 1
        ((attempt++))
    done
    
    log_error "$service_name failed to start after $max_attempts attempts"
    return 1
}

# Function to check and install Python requirements
check_python_requirements() {
    log_info "Checking Python requirements..."
    
    # Check if pip is installed
    if ! command -v pip &> /dev/null; then
        log_error "pip is not installed"
        return 1
    fi
    
    # Install requirements globally without using venv
    log_info "Installing Python requirements globally..."
    pip install -r requirements.txt --user
    if [ $? -ne 0 ]; then
        log_error "Failed to install Python requirements"
        return 1
    fi
    log_success "Python requirements installed successfully"
    
    # Run setup script to ensure models are downloaded
    log_info "Running setup script..."
    python backend/setup.py
    if [ $? -ne 0 ]; then
        log_error "Setup script failed"
        return 1
    fi
    
    return 0
}

# Function to check and install Node.js requirements
check_node_requirements() {
    log_info "Checking Node.js requirements..."
    
    # Check if we're in the frontend directory
    if [ ! -d "frontend" ]; then
        log_error "frontend directory not found"
        return 1
    fi
    
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        log_info "Installing Node.js dependencies..."
        npm install
        if [ $? -ne 0 ]; then
            log_error "Failed to install Node.js dependencies"
            cd ..
            return 1
        fi
        log_success "Node.js dependencies installed successfully"
    else
        log_success "Node.js dependencies already installed"
    fi
    
    cd ..
    return 0
}

# Set up cleanup on script termination
trap cleanup EXIT INT TERM

# Check if ports are already in use
if is_port_in_use 8000; then
    log_warning "Port 8000 is already in use. Stopping existing process..."
    pkill -f "python backend/app.py" || true
    sleep 2
fi

if is_port_in_use 5173; then
    log_warning "Port 5173 is already in use. Stopping existing process..."
    pkill -f "npm run dev" || true
    sleep 2
fi

# Check if Python and npm are available
if ! command -v python &> /dev/null; then
    log_error "Python is not installed"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    log_error "npm is not installed"
    exit 1
fi

# Check and install requirements
check_python_requirements
if [ $? -ne 0 ]; then
    log_error "Failed to set up Python requirements"
    exit 1
fi

check_node_requirements
if [ $? -ne 0 ]; then
    log_error "Failed to set up Node.js requirements"
    exit 1
fi

# Start backend server
log_info "Starting backend server..."
cd "$(dirname "$0")"  # Ensure we're in the right directory
python backend/app.py &
backend_pid=$!

# Wait for backend to be ready
if ! wait_for_service "http://127.0.0.1:8000" "Backend server" 30 "python backend/app.py"; then
    log_error "Failed to start backend server"
    exit 1
fi

# Start frontend server
log_info "Starting frontend server..."
cd frontend && npm run dev &
frontend_pid=$!

# Wait for frontend to be ready
if ! wait_for_service "http://localhost:5173" "Frontend server" 30 "npm run dev"; then
    log_error "Failed to start frontend server"
    exit 1
fi

log_success "Both servers are running!"
log_info "Frontend: http://localhost:5173"
log_info "Backend: http://127.0.0.1:8000"
log_info "Press Ctrl+C to stop both servers"

# Wait for both processes
wait $backend_pid $frontend_pid 