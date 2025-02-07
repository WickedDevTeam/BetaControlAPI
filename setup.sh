#!/bin/bash

# Exit on error
set -e

echo "Starting BetaCensor2 setup..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Install/upgrade pip globally
echo "Upgrading pip..."
pip install --upgrade pip --user

# Install requirements globally
echo "Installing requirements globally..."
pip install -r requirements.txt --user

# Run setup script
echo "Running setup script..."
python backend/setup.py

echo "Setup complete! You can now run the application."
echo "To start the server, run: ./start.sh" 