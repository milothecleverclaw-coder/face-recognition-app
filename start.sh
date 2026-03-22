#!/bin/bash
set -e

echo "================================================"
echo "  Face Recognition App - Build & Start"
echo "================================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build frontend
echo ""
echo "📦 Building frontend..."
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "   Installing dependencies..."
    npm install
fi

echo "   Running Vite build..."
npm run build

# Install backend dependencies
echo ""
echo "🐍 Installing backend dependencies..."
cd ../backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install
source venv/bin/activate
pip install -r requirements.txt --quiet

# Start server
echo ""
echo "🚀 Starting FastAPI server on port 80..."
echo "================================================"
echo ""

# Deactivate venv and run uvicorn directly
deactivate 2>/dev/null || true

# Run uvicorn with the venv python
exec ./venv/bin/uvicorn main:app --host 0.0.0.0 --port 80
