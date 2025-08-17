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


def convert_audio(input_path: str, output_path: str, speed: float = 1.0):
    """
    Convert audio to compatible format and speed up if needed.
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
        model: str = "turbo",
        verbose: Optional[bool] = None,
        temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threshold: Optional[float] = 2.4,
        speed_up: Optional[float] = 1.25,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'\"¿([{-",
        append_punctuations: str = "\"\'.。,，!！?？:：\")]}、",
        clip_timestamps: Union[str, List[float]] = "0",
        hallucination_silence_threshold: Optional[float] = None
):
    # Token validation
    if token not in get_keys():
        logger.warning(f"Invalid token attempt: {token}")
        raise HTTPException(status_code=403, detail="Forbidden")

    model = whisper.load_model(model)  # Load the Whisper model

    logger.info(f"Processing file: {file.filename} with model: {model}")

    # Save uploaded file
    temp_input_path = f"/tmp/input_{file.filename}"
    temp_output_path = f"/tmp/converted_{file.filename}.wav"

    try:
        with open(temp_input_path, "wb") as f:
            f.write(await file.read())

        # Convert audio if needed
        logger.debug("Converting audio file")
        if not convert_audio(temp_input_path, temp_output_path, speed_up):
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
