#!/bin/bash

# Set up Python environment
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"

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
is_running() {
    local pid=$1
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to cleanup processes on exit
cleanup() {
    log_info "Stopping services..."
    if [ -n "$BACKEND_PID" ] && is_running "$BACKEND_PID"; then
        kill "$BACKEND_PID" 2>/dev/null
        log_success "Backend stopped"
    fi
    if [ -n "$FRONTEND_PID" ] && is_running "$FRONTEND_PID"; then
        kill "$FRONTEND_PID" 2>/dev/null
        log_success "Frontend stopped"
    fi
    exit 0
}

# Set up cleanup trap
trap cleanup SIGINT SIGTERM

# Start backend server
log_info "Starting backend server..."
python backend/app.py &
BACKEND_PID=$!

if is_running $BACKEND_PID; then
    log_success "Backend started with PID: $BACKEND_PID"
else
    log_error "Failed to start backend server"
    exit 1
fi

# Check if frontend is already running
if lsof -i :5173 >/dev/null 2>&1; then
    log_warning "Frontend is already running"
else
    # Start frontend server
    log_info "Starting frontend server..."
    cd frontend && npm run dev &
    FRONTEND_PID=$!
    cd ..

    if is_running $FRONTEND_PID; then
        log_success "Frontend started with PID: $FRONTEND_PID"
    else
        log_error "Failed to start frontend server"
        cleanup
        exit 1
    fi
fi

log_success "Services started successfully!"
log_info "Backend running on http://localhost:3000"
log_info "Frontend running on http://localhost:5173"
log_info "API documentation available at http://localhost:3000/docs"
log_info "Press Ctrl+C to stop all services"

# Monitor processes and restart if needed
while true; do
    if ! is_running $BACKEND_PID; then
        log_warning "Backend crashed, restarting..."
        log_info "Starting backend server..."
        python backend/app.py &
        BACKEND_PID=$!
        if is_running $BACKEND_PID; then
            log_success "Backend started with PID: $BACKEND_PID"
        else
            log_error "Failed to restart backend server"
            cleanup
            exit 1
        fi
    fi

    if ! lsof -i :5173 >/dev/null 2>&1 && ! lsof -i :5174 >/dev/null 2>&1; then
        log_warning "Frontend crashed, restarting..."
        log_info "Starting frontend server..."
        cd frontend && npm run dev &
        FRONTEND_PID=$!
        cd ..
        if is_running $FRONTEND_PID; then
            log_success "Frontend started with PID: $FRONTEND_PID"
        else
            log_error "Failed to restart frontend server"
            cleanup
            exit 1
        fi
    fi

    sleep 2
done 