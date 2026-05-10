@echo off
REM Run Music Flamingo FastAPI server with CUDA-enabled PyTorch
REM Uses .venv311 with PyTorch 2.5.1+cu121 and RTX 4070 GPU support

setlocal enabledelayedexpansion

cd /d "d:\Abhijeet\Music Flamingo\flamingo\flamingo"

echo.
echo ========================================
echo Music Flamingo Chatbot - FastAPI Server
echo ========================================
echo.
echo Starting with .venv311 (CUDA-enabled PyTorch)
echo GPU: NVIDIA GeForce RTX 4070 Laptop GPU
echo.

set PRELOAD_MODEL=1
set HF_HOME=%CD%\hf_cache
set HUGGINGFACE_HUB_CACHE=%CD%\hf_cache\hub
set TRANSFORMERS_CACHE=%CD%\hf_cache\hub
".\.venv311\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000

pause
