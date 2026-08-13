#!/bin/bash
# MM-MentalHealth AI - Quick Setup Script
# Usage: bash setup.sh

echo "=========================================="
echo "  MM-MentalHealth AI - Setup Script"
echo "=========================================="

# Check Python version
python3 --version 2>/dev/null || python --version 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found. Please install Python 3.10+ from https://python.org"
    exit 1
fi

echo ""
echo "Step 1: Creating virtual environment..."
python3 -m venv venv || python -m venv venv

echo "Step 2: Activating virtual environment..."
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

echo "Step 3: Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  To run the app:"
echo "    source venv/bin/activate   (macOS/Linux)"
echo "    venv\\Scripts\\activate.bat  (Windows)"
echo "    streamlit run app/app.py"
echo ""
echo "  Then open: http://localhost:8501"
echo "=========================================="
