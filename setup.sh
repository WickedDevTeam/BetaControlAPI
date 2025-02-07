#!/bin/bash

echo "🚀 Setting up BetaControlAPI..."

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p logs
mkdir -p uploads
mkdir -p cache
mkdir -p models

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt

# Download required model files
echo "🔄 Downloading required model files..."
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
    echo "⬇️ Downloading facial landmarks predictor..."
    curl -L "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -o shape_predictor_68_face_landmarks.dat.bz2
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
fi

# Install Node.js dependencies and build frontend
echo "🏗️ Setting up frontend..."
cd frontend
if ! command -v npm &> /dev/null; then
    echo "⚠️ npm not found. Installing Node.js..."
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        brew install node
    elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        # Linux
        curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
fi

echo "📦 Installing frontend dependencies..."
npm install

echo "🏗️ Building frontend for production..."
npm run build

cd ..

# Set up environment variables if not exists
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
fi

# Make start script executable
chmod +x start.sh

echo "✅ Setup completed successfully!"
echo "🚀 Run './start.sh' to start the application" 