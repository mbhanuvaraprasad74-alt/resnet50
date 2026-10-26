@echo off
title Setup PulmoScan AI Environment
echo ======================================================================
echo    PulmoScan AI - Automatic Environment Setup
echo ======================================================================
echo.

python --version
if errorlevel 1 (
    echo [!] Python is not detected in your PATH.
    echo [*] Please install Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo [*] IMPORTANT: Ensure you check "Add python.exe to PATH" during installation.
    pause
    exit /b
)

echo [*] Creating Python virtual environment 'venv'...
python -m venv venv

echo [*] Activating virtual environment...
call venv\Scripts\activate.bat

echo [*] Upgrading pip...
python -m pip install --upgrade pip

echo [*] Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo [✓] Environment setup completed successfully!
echo [*] You can now run the app using 'run_app.bat' or 'streamlit run app.py'
pause
