#!/bin/bash

echo "🚀 Setting up BetaControlAPI..."

# Create necessary directories
echo "📁 Creating required directories..."
mkdir -p logs
mkdir -p uploads
mkdir -p cache
mkdir -p models

# Basic pip setup
echo "📦 Setting up pip..."
python3 -m pip install --upgrade pip

# Install basic requirements first
echo "📦 Installing basic requirements..."
pip install wheel setuptools numpy

# Install dlib separately
echo "📦 Installing dlib..."
pip install dlib

# Install other dependencies
echo "📦 Installing other dependencies..."
pip install flask flask-cors pillow opencv-python requests psutil nudenet

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