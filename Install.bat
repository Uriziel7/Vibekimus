@echo off
cd /d "%~dp0"
if not exist "venv" (
    py -3.12 -m venv venv
    if errorlevel 1 (
        py -3.11 -m venv venv
    )
    if errorlevel 1 (
        py -3.10 -m venv venv
    )
    if errorlevel 1 (
        python -m venv venv
    )
    if errorlevel 1 (
        echo No compatible Python found! Please install Python 3.10, 3.11, or 3.12 from python.org
        pause
        exit
    )
)
rem The venv's own Python, not activate.bat or pip.exe: those keep the path the venv was made in.
set "PYTHON=%~dp0venv\Scripts\python.exe"
"%PYTHON%" -m pip install -r requirements.txt
"%PYTHON%" -m pip install --no-deps -r requirements-ocr.txt
"%PYTHON%" Vibekimus.py
pause
