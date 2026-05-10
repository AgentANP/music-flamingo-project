@echo off
REM Stop the Music Flamingo FastAPI server by killing the process listening on port 8000

setlocal enabledelayedexpansion

set PORT=8000
set FOUND=0

for /f "tokens=5" %%P in ('netstat -ano ^| findstr ":%PORT%" ^| findstr LISTENING') do (
    set FOUND=1
    echo Stopping PID %%P listening on port %PORT%...
    taskkill /F /PID %%P
)

if %FOUND%==0 (
    echo No listening process found on port %PORT%.
)