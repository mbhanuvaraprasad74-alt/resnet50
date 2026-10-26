@echo off
title PulmoScan AI - ResNet-50 Diagnostic Web Application
echo ======================================================================
echo    PulmoScan AI - ResNet50 Radiographic Diagnostic Web Application
echo    Candidate: M. Akhil Kumar (25G31D5819) | SJCET CSE
echo ======================================================================
echo.

:: Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [*] Activating virtual environment...
    call venv\Scripts\activate.bat
)

echo [*] Launching Streamlit Web Diagnostic Interface...
if exist "venv\Scripts\streamlit.exe" (
    venv\Scripts\streamlit.exe run app.py
) else (
    streamlit run app.py
)

pause
