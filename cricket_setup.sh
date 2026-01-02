#!/bin/bash
# Cricket Ball Detection Setup Script
# EdgeFleet.AI Assessment - IIT BHU

echo "=========================================="
echo "Cricket Ball Detection Setup"
echo "=========================================="

# Create directory structure
echo "Creating directory structure..."
mkdir -p code
mkdir -p annotations
mkdir -p results
mkdir -p runs/train

# Check Python version
echo ""
echo "Checking Python version..."
python --version

# Create virtual environment (optional but recommended)
echo ""
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python -m venv venv
    echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install ultralytics>=8.0.0
pip install opencv-python>=4.8.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install torch>=2.0.0
pip install torchvision>=0.15.0
pip install matplotlib>=3.7.0
pip install tqdm>=4.65.0

# Verify installations
echo ""
echo "Verifying installations..."
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "from ultralytics import YOLO; print('Ultralytics: OK')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"

# Download YOLOv8 pretrained weights
echo ""
echo "Downloading YOLOv8 nano weights..."
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Download cricket ball dataset from Roboflow/Kaggle"
echo "2. Configure data.yaml with your dataset path"
echo "3. Run training: python code/train.py"
echo "4. Download test videos from provided Google Drive link"
echo "5. Run inference: python code/inference.py --video test.mp4 --model best.pt"
echo ""
echo "For help: python code/inference.py --help"
echo "=========================================="
