@echo off
rem Runs the program with the venv's own Python (works wherever the folder is moved).
set "PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo Venv not found! Please run Install.bat first.
    pause
    exit
)
cd /d "%~dp0"
"%PYTHON%" Vibekimus.py
pause
