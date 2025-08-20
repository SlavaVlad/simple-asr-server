import logging
import os
import subprocess
import tempfile
from typing import Optional
from enum import Enum

import whisper
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query
from fastapi.security import APIKeyHeader
from fastapi.responses import PlainTextResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Simple ASR Server", description="Audio transcription API using Whisper")

# API key header
api_key_header = APIKeyHeader(name="x-api-key")

# Global model variable
default_model = None

class OutputFormat(str, Enum):
    plaintext = "plaintext"
    simple = "simple"
    json = "json"

def get_keys():
    keys_file = os.getenv("KEYS_FILE", "keys.txt")
    if not os.path.exists(keys_file):
        # Create a new keys file with a default key
        default_key = os.urandom(32).hex()
        with open(keys_file, "w") as f:
            f.write(default_key + "\n")
        logger.info(f"Created new keys file with default key: {default_key}")
        return [default_key]
    else:
        # Read keys from the existing file
        with open(keys_file, "r") as f:
            keys = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(keys)} keys from file")
        if not keys:
            raise ValueError("No keys found in keys.txt")
        return keys

def load_default_model():
    """Load the default model on startup"""
    global default_model
    model_name = os.getenv("DEFAULT_MODEL", "turbo")
    model_download_root = os.getenv("MODEL_DOWNLOAD_ROOT", None)

    logger.info(f"Loading default model: {model_name}")
    try:
        default_model = whisper.load_model(model_name, download_root=model_download_root, in_memory=True)
        logger.info(f"Successfully loaded model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load default model {model_name}: {e}")
        raise

def get_model(model_name: Optional[str] = None):
    """Get model - either default or load new one if specified"""
    global default_model

    if model_name is None:
        return default_model

    # If different model requested, load it
    if model_name != os.getenv("DEFAULT_MODEL", "turbo"):
        model_download_root = os.getenv("MODEL_DOWNLOAD_ROOT", None)
        logger.info(f"Loading requested model: {model_name}")
        return whisper.load_model(model_name, download_root=model_download_root)

    return default_model

def convert_audio(input_path: str, output_path: str, speed: float = 1.0):
    """Convert audio to compatible format and speed up if needed."""
    try:
        command = [
            'ffmpeg', '-i', input_path,
            '-filter:a', f'atempo={speed}',
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_path,
            '-y'
        ]
        logger.debug(f"Running FFmpeg command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr}")
        return False

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    token: str = Depends(api_key_header),
    model_name: Optional[str] = Query(None, description="Model name to use for transcription"),
    output_format: OutputFormat = Query(OutputFormat.json, description="Output format: plaintext, simple, or json"),
    speedup: float = Query(1.0, ge=0.25, le=4.0, description="Speed up factor for audio (0.25-4.0)")
):
    """Transcribe audio file with configurable output format"""

    # Token validation
    if token not in get_keys():
        logger.warning(f"Invalid token attempt: {token}")
        raise HTTPException(status_code=403, detail="Forbidden")

    logger.info(f"Processing file: {file.filename}, model: {model_name or 'default'}, format: {output_format}, speedup: {speedup}")

    # Get model
    try:
        model = get_model(model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Create temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_input:
        temp_input_path = temp_input.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_output:
        temp_output_path = temp_output.name

    try:
        # Save uploaded file
        with open(temp_input_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Convert audio if speedup is not 1.0 or format needs conversion
        if speedup != 1.0 or not file.filename.lower().endswith('.wav'):
            logger.debug(f"Converting audio file with speedup: {speedup}")
            if not convert_audio(temp_input_path, temp_output_path, speedup):
                raise HTTPException(status_code=400, detail="Audio conversion failed")
            audio_file_path = temp_output_path
        else:
            audio_file_path = temp_input_path

        # Transcribe
        logger.info("Starting transcription")
        result = model.transcribe(audio_file_path)

        # Format output based on requested format
        if output_format == OutputFormat.plaintext:
            return PlainTextResponse(content=result["text"], media_type="text/plain")
        elif output_format == OutputFormat.simple:
            return {"text": result["text"]}
        else:  # json format
            return result

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary files
        for path in [temp_input_path, temp_output_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {path}: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": default_model is not None}

def main():
    import uvicorn

    # Load default model and keys
    load_default_model()
    get_keys()

    port = int(os.getenv("PORT", 9854))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()
