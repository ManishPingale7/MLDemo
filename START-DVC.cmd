@echo off
cd /d "%~dp0"
where git >nul 2>nul
if errorlevel 1 (
  echo Install Git for Windows first.
  pause
  exit /b 1
)
if not exist .venv-dvc\Scripts\python.exe py -3.11 -m venv .venv-dvc
if errorlevel 1 goto failed
.venv-dvc\Scripts\python.exe -m pip install -r requirements-dvc.txt
if errorlevel 1 goto failed
.venv-dvc\Scripts\python.exe scripts\dvc_demo.py
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo DVC setup or demonstration failed. Read the error above.
pause
exit /b 1
