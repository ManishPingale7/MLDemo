@echo off
setlocal
cd /d "%~dp0"
where git >nul 2>nul
if errorlevel 1 (
  echo Install Git for Windows first, then reopen this file.
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 goto :fail
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail
.venv\Scripts\python.exe scripts\run_all.py
if errorlevel 1 goto :fail
.venv\Scripts\python.exe -m jupyter lab notebooks
exit /b %errorlevel%
:fail
 echo Setup or execution failed. Read the error above.
 pause
 exit /b 1
