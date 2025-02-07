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

# Function to start the backend
start_backend() {
    echo "🚀 Starting backend server..."
    if [ "$PRODUCTION" = "true" ]; then
        # Production mode with gunicorn
        gunicorn --bind 0.0.0.0:${PORT:-3000} \
                 --workers 4 \
                 --threads 2 \
                 --timeout 120 \
                 --access-logfile logs/access.log \
                 --error-logfile logs/error.log \
                 --capture-output \
                 --enable-stdio-inheritance \
                 "backend.app:app" &
    else
        # Development mode
        python3 backend/app.py &
    fi
    BACKEND_PID=$!
    echo "📝 Backend PID: $BACKEND_PID"
}

# Function to start the frontend
start_frontend() {
    echo "🌐 Starting frontend server..."
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
    echo "📝 Frontend PID: $FRONTEND_PID"
}

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if services are already running
if check_process "backend/app.py" || check_process "gunicorn"; then
    echo "⚠️ Backend is already running"
else
    start_backend
fi

if check_process "npm run dev" || check_process "serve -s dist"; then
    echo "⚠️ Frontend is already running"
else
    start_frontend
fi

# Function to cleanup processes on exit
cleanup() {
    echo "🛑 Stopping services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

# Register cleanup function
trap cleanup SIGINT SIGTERM

echo "✅ Services started successfully!"
echo "📊 Backend running on http://localhost:${PORT:-3000}"
echo "🌐 Frontend running on http://localhost:${FRONTEND_PORT:-5173}"
echo "📚 API documentation available at http://localhost:${PORT:-3000}/docs"
echo "💡 Press Ctrl+C to stop all services"

# Keep script running and monitor processes
while true; do
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "⚠️ Backend crashed, restarting..."
        start_backend
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "⚠️ Frontend crashed, restarting..."
        start_frontend
    fi
    sleep 5
done 