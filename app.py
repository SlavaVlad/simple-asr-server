import logging
import os
import subprocess
import tempfile
from typing import Optional, Union, List, Tuple
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
    speedup: float = Query(1.0, ge=0.25, le=4.0, description="Speed up factor for audio (0.25-4.0)"),
    # Whisper model parameters
    verbose: Optional[bool] = Query(None, description="Whether to print out the progress and debug messages"),
    temperature: Union[float, str] = Query("0.0,0.2,0.4,0.6,0.8,1.0", description="Temperature for sampling (single float or comma-separated values)"),
    compression_ratio_threshold: Optional[float] = Query(2.4, description="If the gzip compression ratio is above this value, treat as failed"),
    logprob_threshold: Optional[float] = Query(-1.0, description="If the average log probability over sampled tokens is below this value, treat as failed"),
    no_speech_threshold: Optional[float] = Query(0.6, description="If the no_speech probability is higher than this value AND the average log probability over sampled tokens is below logprob_threshold, consider the segment as silent"),
    condition_on_previous_text: bool = Query(True, description="If True, the previous output of the model is provided as a prompt for the next window"),
    initial_prompt: Optional[str] = Query(None, description="Optional text to provide as a prompt for the first window"),
    carry_initial_prompt: bool = Query(False, description="If True, the initial prompt is carried over to the next window"),
    word_timestamps: bool = Query(False, description="Extract word-level timestamps using the cross-attention pattern and dynamic time warping"),
    prepend_punctuations: str = Query("\"'([{-", description="If word_timestamps is True, merge these punctuation marks with the next word"),
    append_punctuations: str = Query("\"'.,:;!?)]}", description="If word_timestamps is True, merge these punctuation marks with the previous word"),
    clip_timestamps: Union[str, List[float]] = Query("0", description="Comma-separated list of clip timestamps to use for transcription"),
    hallucination_silence_threshold: Optional[float] = Query(None, description="When word_timestamps is True, skip silent periods longer than this threshold (in seconds)"),
):
    """Transcribe audio file with configurable output format and comprehensive Whisper parameters"""

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

        # Prepare transcription parameters
        transcribe_params = {}

        # Handle temperature parameter (can be single value or tuple)
        if isinstance(temperature, str) and "," in temperature:
            try:
                temp_values = [float(x.strip()) for x in temperature.split(",")]
                transcribe_params["temperature"] = tuple(temp_values)
            except ValueError:
                transcribe_params["temperature"] = 0.0
        else:
            try:
                transcribe_params["temperature"] = float(temperature)
            except (ValueError, TypeError):
                transcribe_params["temperature"] = 0.0

        # Handle clip_timestamps parameter
        if isinstance(clip_timestamps, str) and clip_timestamps != "0":
            try:
                if "," in clip_timestamps:
                    transcribe_params["clip_timestamps"] = [float(x.strip()) for x in clip_timestamps.split(",")]
                else:
                    transcribe_params["clip_timestamps"] = clip_timestamps
            except ValueError:
                transcribe_params["clip_timestamps"] = "0"
        else:
            transcribe_params["clip_timestamps"] = clip_timestamps

        # Add other parameters if they are not None
        if verbose is not None:
            transcribe_params["verbose"] = verbose
        if compression_ratio_threshold is not None:
            transcribe_params["compression_ratio_threshold"] = compression_ratio_threshold
        if logprob_threshold is not None:
            transcribe_params["logprob_threshold"] = logprob_threshold
        if no_speech_threshold is not None:
            transcribe_params["no_speech_threshold"] = no_speech_threshold

        transcribe_params["condition_on_previous_text"] = condition_on_previous_text
        transcribe_params["carry_initial_prompt"] = carry_initial_prompt
        transcribe_params["word_timestamps"] = word_timestamps
        transcribe_params["prepend_punctuations"] = prepend_punctuations
        transcribe_params["append_punctuations"] = append_punctuations

        if initial_prompt is not None:
            transcribe_params["initial_prompt"] = initial_prompt
        if hallucination_silence_threshold is not None:
            transcribe_params["hallucination_silence_threshold"] = hallucination_silence_threshold

        # Transcribe
        logger.info("Starting transcription")
        logger.debug(f"Transcription parameters: {transcribe_params}")
        result = model.transcribe(audio_file_path, **transcribe_params)

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
    return {"status": "healthy", "model_loaded": default_model is not None, "model_name": default_model.__str__()}

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
