#!/usr/bin/env bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo "Script directory: $SCRIPT_DIR"

# Define the virtual environment directory name within the script directory
VENV_DIR="$SCRIPT_DIR/venv"
echo "Virtual environment target: $VENV_DIR"

# Create virtual environment if the activate script doesn't exist
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "--------------------------------------------------"
        echo "ERROR: Failed to create virtual environment."
        echo "Ensure python3 and python3-venv package are installed."
        echo "--------------------------------------------------"
        exit 1
    fi
else
    echo "Virtual environment found."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "--------------------------------------------------"
    echo "ERROR: Failed to activate virtual environment."
    echo "--------------------------------------------------"
    exit 1
fi

# Ensure pip is up-to-date within the venv
echo "Updating pip..."
pip install --upgrade pip

# Crucial Step: Remind user about PyTorch BEFORE installing others
echo "--------------------------------------------------"
echo "IMPORTANT: This script assumes PyTorch (appropriate CPU/CUDA version)"
echo "has ALREADY been installed in the '$VENV_DIR' environment."
echo "If not, please stop this script (Ctrl+C), install PyTorch manually:"
echo "   source \"$VENV_DIR/bin/activate\""
echo "   pip install torch torchvision torchaudio <...options based on your system...>"
echo "   (See: https://pytorch.org/get-started/locally/)"
echo "and then re-run this script."
echo "--------------------------------------------------"
# Optional: Add a short pause/prompt?
# read -p "Press Enter to continue if PyTorch is installed, or Ctrl+C to exit..."

# Install other dependencies from requirements.txt
echo "Installing other dependencies (PyQt5, safetensors, tqdm)..."
# Ensure we are in script dir to find requirements.txt reliably
cd "$SCRIPT_DIR"
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "--------------------------------------------------"
    echo "ERROR: Failed to install dependencies from requirements.txt."
    echo "Check the file contents and your internet connection."
    echo "--------------------------------------------------"
    exit 1
fi

# Launch the GUI
echo "Starting Flux LoRA Merger GUI..."
python3 flux_lora_merger_gui.py
if [ $? -ne 0 ]; then
    echo "--------------------------------------------------"
    echo "ERROR: Failed to run the Python script."
    echo "Check the script for errors or missing dependencies (like PyTorch)."
    echo "--------------------------------------------------"
    exit 1
fi

echo "--------------------------------------------------"
echo "GUI closed. Script finished."
echo "To deactivate the virtual environment manually, type: deactivate"
echo "--------------------------------------------------"

exit 0