@echo off
cd /d "%~dp0"

set VENV_DIR=venv

echo Checking Python...
python --version || (echo ERROR: Python not found. Ensure it's installed/in PATH. & pause & exit /b 1)

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%" || (echo ERROR: Failed to create venv. & pause & exit /b 1)
) else (
    echo Virtual environment found.
)

echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat" || (echo ERROR: Failed to activate venv. & pause & exit /b 1)

echo Updating pip...
python -m pip install --upgrade pip || (echo ERROR: Failed to update pip. & pause & exit /b 1)

echo Checking PyTorch...
pip show torch >nul 2>&1
if errorlevel 1 (
    echo Installing PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu || (echo ERROR: Failed to install PyTorch. & pause & exit /b 1)
) else (
    echo PyTorch already installed.
)

if exist requirements.txt (
    echo Installing requirements.txt...
    pip install -r requirements.txt || (echo ERROR: Failed to install dependencies. & pause & exit /b 1)
) else (
    echo ERROR: requirements.txt missing. & pause & exit /b 1
)

if exist flux_lora_merger_gui.py (
    echo Running GUI...
    python flux_lora_merger_gui.py || (echo ERROR: GUI script failed. & pause & exit /b 1)
) else (
    echo ERROR: flux_lora_merger_gui.py missing. & pause & exit /b 1
)

echo Script finished.
pause
exit /b 0