@echo off
rem Builds dist\Vibekimus.exe (about 15 MB). The program downloads its libraries and
rem models itself on first start. Needs the venv made by Install.bat (with PyInstaller).
cd /d "%~dp0"
venv\Scripts\python tools\build_exe.py
pause
