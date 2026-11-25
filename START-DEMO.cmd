@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  py -3.11 -m venv .venv
  if errorlevel 1 goto :fail
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto :fail
.venv\Scripts\python.exe scripts\run_demo.py
if errorlevel 1 goto :fail
exit /b 0
:fail
 echo Demo setup or execution failed. Read the error above.
 pause
 exit /b 1
