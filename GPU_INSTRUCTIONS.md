GPU setup and PyTorch installation (Windows, RTX 40xx)

1) Verify NVIDIA driver
- Make sure you have a recent NVIDIA driver installed (>= 535 recommended for CUDA 12.x).
- Check in Device Manager -> Display adapters, or run `nvidia-smi` in a terminal (should show your GPU).

2) Install CUDA-enabled PyTorch wheel (recommended)
- Choose a wheel matching your CUDA runtime preference. Examples:

  CUDA 12.1 (recommended for recent drivers):

  ```powershell
  pip install --upgrade pip
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

- Install the audio loader dependency used by track analysis:

  ```powershell
  pip install librosa
  ```

  CUDA 11.8 (if you need older runtime):

  ```powershell
  pip install --upgrade pip
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

- If you prefer CPU-only PyTorch (no GPU):

  ```powershell
  pip install --upgrade pip
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

3) Verify installation and CUDA availability
- Run this Python snippet in the same environment used to run `main.py`:

  ```powershell
  python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda_version', torch.version.cuda); print('device_count', torch.cuda.device_count()); print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
  ```

- Expected: `cuda_available` should be `True` and `device_name` should show `NVIDIA GeForce RTX 4070` (or similar).
- If you just installed `librosa`, restart the runtime/server so the new package is visible to the running process.

4) Windows page file (virtual memory)
- With aggressive offloading you may need a larger page file. If you get OSError with Windows winerror 1455, increase Virtual Memory: System Properties > Advanced > Performance > Advanced > Virtual memory. Recommended: Initial 32768 MB, Maximum 65536 MB.

5) Notes
- The model in this repo tries to auto-detect CUDA. If `torch` is missing or is CPU-only, install a CUDA-enabled wheel into the same Python environment that runs `uvicorn`/`main.py`.
- If you use a virtual environment, activate it before installing packages and before starting the server.
- Audio analysis requires `librosa`; install it in the same environment as the server.

6) Stop the server
- If the launcher window is closed but the server is still running, run `STOP_SERVER.bat` from this folder.
- It stops whatever process is listening on port `8000`.

If you want, I can try installing a CUDA-enabled PyTorch wheel into the active environment and re-run the diagnostics.
