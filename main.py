"""
Music Flamingo Chatbot - FastAPI Backend
Model: nvidia/music-flamingo-2601-hf

Install dependencies:
    pip install fastapi uvicorn python-multipart
    pip install transformers accelerate safetensors
    pip install torch torchaudio

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Music Flamingo Chatbot", version="1.0.0")

# Allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model state ────────────────────────────────────────────────────────────────
model = None
processor = None
MODEL_ID = "nvidia/music-flamingo-2601-hf"
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "0") == "1"
TEMP_DIR = Path(tempfile.gettempdir()) / "music_flamingo"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    """Load Music Flamingo model and processor."""
    global model, processor
    if model is not None:
        return

    logger.info(f"Loading model: {MODEL_ID}")
    try:
        from transformers import MusicFlamingoForConditionalGeneration, AutoProcessor
        try:
            import torch
        except Exception as e:
            raise RuntimeError(
                "PyTorch is not installed in this Python environment.\n"
                "Install a CUDA-enabled PyTorch to use your NVIDIA GPU (example for CUDA 12.1):\n"
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n"
                "Or install CPU-only PyTorch:\n"
                "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
                f"Original error: {e}"
            )

        # Log PyTorch / CUDA diagnostics
        try:
            cuda_ok = torch.cuda.is_available()
            logger.info(f"PyTorch version: {getattr(torch, '__version__', 'n/a')}; cuda_available: {cuda_ok}")
            if cuda_ok:
                try:
                    dev_name = torch.cuda.get_device_name(0)
                except Exception:
                    dev_name = "(unknown)"
                logger.info(f"CUDA device: {dev_name}")
        except Exception:
            logger.info("Could not query CUDA status from PyTorch.")

        processor = AutoProcessor.from_pretrained(MODEL_ID)

        # Choose device mapping and dtypes depending on CUDA availability
        use_cuda = False
        try:
            use_cuda = torch.cuda.is_available()
        except Exception:
            use_cuda = False

        if use_cuda:
            # For CUDA: use device_map="auto" for intelligent memory distribution
            # The meta device warning is accelerate's way of saying it's using offloading strategy
            # This is normal and expected behavior for large models on 8GB GPUs
            device_map = "auto"
            torch_dtype = torch.float16
            max_memory = {0: "7GB", "cpu": "16GB"}
        else:
            logger.warning(
                "CUDA is not available. The model will run on CPU.\n"
                "If you expect to use your NVIDIA GPU (RTX 4070), install a CUDA-enabled PyTorch wheel and ensure NVIDIA drivers are up to date.\n"
                "See GPU_INSTRUCTIONS.md for commands."
            )
            device_map = {"": "cpu"}
            torch_dtype = torch.float32
            max_memory = None

        model = MusicFlamingoForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map=device_map,
            torch_dtype=torch_dtype,
            offload_folder="offload",
            offload_state_dict=True,
            low_cpu_mem_usage=True,
            max_memory=max_memory,
        )
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)

        # Windows OSError 1455: paging file too small for loading/offloading tensors.
        if isinstance(e, OSError) and getattr(e, "winerror", None) == 1455:
            raise RuntimeError(
                "Model load failed due to low Windows virtual memory (paging file), "
                "not because of model weights or GPU detection.\n"
                "Fix: increase the page file size and restart Windows.\n"
                "Recommended for this model on 8GB VRAM: set paging file to at least 64 GB "
                "(Initial 32768 MB, Maximum 65536 MB).\n"
                "Path: System Properties > Advanced > Performance Settings > Advanced > "
                "Virtual memory.\n"
                f"Original error: {e}"
            )

        raise RuntimeError(
            f"Could not load {MODEL_ID}. Ensure dependencies are installed:\n"
            "  pip install transformers accelerate safetensors\n"
            "And install PyTorch (see GPU_INSTRUCTIONS.md for CUDA-enabled wheels).\n"
            f"Original error: {e}"
        )


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    if not PRELOAD_MODEL:
        logger.info("Skipping model preload at startup (lazy load enabled).")
        return

    try:
        load_model()
    except Exception as e:
        logger.warning(f"Model not loaded on startup: {e}")


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_id": MODEL_ID,
    }


# ── Caption / Generate endpoint ───────────────────────────────────────────────
@app.post("/generate")
async def generate(
    audio: UploadFile = File(...),
    prompt: str = Form(
        default="Write a rich caption describing this music track — include genre, tempo, "
                "key, instruments, production style, and the overall mood it creates."
    ),
    max_new_tokens: int = Form(default=512),
):
    """
    Upload an audio file and a text prompt.
    Returns a generated description/caption from Music Flamingo.
    """
    if model is None:
        try:
            load_model()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

    # Validate audio format
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    suffix = Path(audio.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    # Save upload to temp file
    tmp_path = TEMP_DIR / f"upload_{os.urandom(8).hex()}{suffix}"
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": str(tmp_path)},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        # Move tensors to model device
        import torch
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Cast input_features to model dtype if present
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(
                next(model.parameters()).dtype
            )

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        decoded = processor.batch_decode(
            outputs[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        caption = decoded[0].strip() if decoded else ""
        return JSONResponse({"caption": caption, "prompt": prompt})

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


# ── Serve frontend ─────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
