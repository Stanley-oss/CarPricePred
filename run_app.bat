@echo off
cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    echo [Backend] Found virtual environment 'venv', activating...
    start "CarPricePred Backend" cmd /k "call venv\Scripts\activate && python -m backend.main"
) else if exist ".venv\Scripts\activate.bat" (
    echo [Backend] Found virtual environment '.venv', activating...
    start "CarPricePred Backend" cmd /k "call .venv\Scripts\activate && python -m backend.main"
) else (
    echo [Backend] WARNING: No virtual environment found! Trying global python...
    start "CarPricePred Backend" cmd /k "python -m backend.main"
)

timeout /t 2 /nobreak >nul

if exist "frontend" (
    echo [Frontend] Launching React App...
    cd frontend
    start "CarPricePred Frontend" cmd /k "npm run dev"
) else (
    echo [Error] 'frontend' folder not found!
    pause
)

exit