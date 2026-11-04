@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>&1
if not errorlevel 1 (
    py -3 start.py %*
) else (
    python start.py %*
)
if errorlevel 1 (
    echo Startup failed. Install Python 3.11+ or fix the error above.
    pause
    exit /b 1
)
