#!/bin/bash
# Setup script for Medical NLP Pipeline

echo "=========================================="
echo "Medical NLP Pipeline - Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing Python packages..."
pip install -r requirements.txt

# Download spaCy models
echo ""
echo "Downloading spaCy models..."
python -m spacy download en_core_web_sm

# Optional: Download medical model
echo ""
read -p "Download medical spaCy model (en_core_sci_md)? This improves accuracy but takes ~100MB. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading medical model..."
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz
fi

# Create output directory
echo ""
echo "Creating output directory..."
mkdir -p output

# Test installation
echo ""
echo "Testing installation..."
python -c "
import spacy
import transformers
import torch
print('✓ All core packages imported successfully!')
print(f'✓ PyTorch version: {torch.__version__}')
print(f'✓ Transformers version: {transformers.__version__}')
print(f'✓ spaCy version: {spacy.__version__}')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To run the pipeline:"
echo "  1. Activate the virtual environment:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "     venv\\Scripts\\activate"
else
    echo "     source venv/bin/activate"
fi
echo "  2. Run the pipeline:"
echo "     python pipeline.py"
echo ""
echo "For more information, see README.md"
echo ""
