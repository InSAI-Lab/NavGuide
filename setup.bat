@echo off
setlocal
cd /d "%~dp0"
py -3.11 -m venv .venv
if errorlevel 1 exit /b 1
.venv\Scripts\python -m pip install -r requirements.txt -c constraints.txt
if errorlevel 1 exit /b 1
if not exist .env copy .env.example .env
echo Ready: .venv\Scripts\python -m navguide
