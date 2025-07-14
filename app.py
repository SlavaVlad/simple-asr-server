import logging
import os
import subprocess
import time
from os import getenv
from typing import Dict

import gigaam
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.security import APIKeyHeader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

model = gigaam.load_model("v2_ctc", device=getenv("ASR_DEVICE"), download_root=getenv("ASR_MODELS_ROOT"))

# API key header
api_key_header = APIKeyHeader(name="x-api-key")


def get_keys():  # не бейте меня за это
    keys_file = "keys.txt"
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
        logger.debug(f"Keys: {keys}")
        if not keys:
            raise ValueError("No keys found in keys.txt")
        return keys


def convert_audio(input_path: str, output_path: str, speed: float = 1.25):
    """
    Convert audio to compatible format and speed up
    """
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
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        return False


class TranscriptionMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.end_time = None
        self.text_length = 0
        self.audio_duration = 0

    def stop(self, text: str, audio_duration: float):
        self.end_time = time.time()
        self.text_length = len(text)
        self.audio_duration = audio_duration

    def get_metrics(self) -> Dict[str, float]:
        processing_time = self.end_time - self.start_time
        return {
            "processing_time_seconds": round(processing_time, 2),
            "characters_per_second": round(self.text_length / processing_time, 2),
            "audio_realtime_ratio": round(self.audio_duration / processing_time, 2),
            "audio_duration": round(self.audio_duration, 2),
            "text_length": self.text_length
        }


def get_audio_duration(file_path: str) -> float:
    """Get audio duration using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        file_path
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)
    except:
        return 0.0


@app.post("/transcribe")
async def transcribe_audio(
        file: UploadFile = File(...),
        token: str = Depends(api_key_header),
        model_name: str = "turbo"
):
    # Token validation
    if token not in get_keys():
        logger.warning(f"Invalid token attempt: {token}")
        raise HTTPException(status_code=403, detail="Forbidden")

    logger.info(f"Processing file: {file.filename} with model: {model_name}")
    metrics = TranscriptionMetrics()

    # Save uploaded file
    temp_input_path = f"/tmp/input_{file.filename}"
    temp_output_path = f"/tmp/converted_{file.filename}.wav"

    try:
        with open(temp_input_path, "wb") as f:
            f.write(await file.read())

        # Convert audio if needed
        logger.debug("Converting audio file")
        if not convert_audio(temp_input_path, temp_output_path):
            raise HTTPException(status_code=400, detail="Audio conversion failed")

        # Get audio duration before speed up
        original_duration = get_audio_duration(temp_input_path)

        # Transcribe
        logger.info("Starting transcription")
        if original_duration > 30:
            logger.info("Audio duration > 30 seconds, using transcribe_longform")
            transcription_result = model.transcribe_longform(
                temp_output_path
            )
        else:
            logger.info("Audio duration <= 30 seconds, using transcribe")
            transcription_result = model.transcribe(
                temp_output_path
            )

        full_text = ""
        for part in transcription_result:
            if part["transcription"].strip() != "":
                full_text += part["transcription"].strip() + " "

        result = {
            "transcription": transcription_result,
            "text": full_text
        }

        # Calculate metrics
        metrics.stop(full_text, original_duration)
        logger.info(f"Transcription metrics: {metrics.get_metrics()}")

        # Add metrics to result
        result["metrics"] = metrics.get_metrics()

        return result

    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temporary files
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


def main():
    import uvicorn
    get_keys()
    uvicorn.run(app, host="0.0.0.0", port=9854, log_level="debug")

if __name__ == "__main__":
    main()
