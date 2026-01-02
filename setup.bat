@echo off
REM Setup script for Medical NLP Pipeline (Windows)

echo ==========================================
echo Medical NLP Pipeline - Setup Script
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo Installing Python packages...
pip install -r requirements.txt
echo.

REM Download spaCy models
echo Downloading spaCy models...
python -m spacy download en_core_web_sm
echo.

REM Optional: Download medical model
set /p download_medical="Download medical spaCy model (en_core_sci_md)? This improves accuracy but takes ~100MB. (y/n): "
if /i "%download_medical%"=="y" (
    echo Downloading medical model...
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz
)
echo.

REM Create output directory
echo Creating output directory...
if not exist "output" mkdir output
echo.

REM Test installation
echo Testing installation...
python -c "import spacy; import transformers; import torch; print('✓ All core packages imported successfully!')"
echo.

echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo To run the pipeline:
echo   1. Activate the virtual environment:
echo      venv\Scripts\activate
echo   2. Run the pipeline:
echo      python pipeline.py
echo.
echo For more information, see README.md
echo.

pause
