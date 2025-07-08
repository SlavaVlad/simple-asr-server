import os
import tempfile
import sys
import yaml
from typing import Optional, List, Union, Tuple, Iterable
from fastapi import FastAPI, UploadFile, File, Depends
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel
from converter import convert_to_wav

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

app = FastAPI()

w_config = config['whisper']

class TranscriptionOptions(BaseModel):
    language: Optional[str] = w_config.get('language')
    task: str = w_config.get('task', 'transcribe')
    beam_size: int = w_config.get('beam_size', 5)
    best_of: int = w_config.get('best_of', 5)
    patience: float = w_config.get('patience', 1.0)
    length_penalty: float = w_config.get('length_penalty', 1.0)
    repetition_penalty: float = w_config.get('repetition_penalty', 1.0)
    no_repeat_ngram_size: int = w_config.get('no_repeat_ngram_size', 0)
    temperature: Union[float, List[float], Tuple[float, ...]] = w_config.get('temperature', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    log_progress: bool = w_config.get('log_progress', False)
    compression_ratio_threshold: Optional[float] = w_config.get('compression_ratio_threshold', 2.4)
    log_prob_threshold: Optional[float] = w_config.get('log_prob_threshold', -1.0)
    no_speech_threshold: Optional[float] = w_config.get('no_speech_threshold', 0.6)
    condition_on_previous_text: bool = w_config.get('condition_on_previous_text', True)
    prompt_reset_on_temperature: float = w_config.get('prompt_reset_on_temperature', 0.5)
    initial_prompt: Optional[Union[str, Iterable[int]]] = w_config.get('initial_prompt')
    prefix: Optional[str] = w_config.get('prefix')
    suppress_blank: bool = w_config.get('suppress_blank', True)
    suppress_tokens: Optional[List[int]] = w_config.get('suppress_tokens', [-1])
    without_timestamps: bool = w_config.get('without_timestamps', False)
    max_initial_timestamp: float = w_config.get('max_initial_timestamp', 1.0)
    word_timestamps: bool = w_config.get('word_timestamps', False)
    prepend_punctuations: str = w_config.get('prepend_punctuations', '"\'“¿([{-')
    append_punctuations: str = w_config.get('append_punctuations', '"\'.。,，!！?？:：”)]}、')
    vad_filter: bool = w_config.get('vad_filter', False)
    vad_parameters: Optional[dict] = w_config.get('vad_parameters')
    max_new_tokens: Optional[int] = w_config.get('max_new_tokens')
    chunk_length: Optional[int] = w_config.get('chunk_length')
    clip_timestamps: Union[str, List[float]] = w_config.get('clip_timestamps', "0")
    hallucination_silence_threshold: Optional[float] = w_config.get('hallucination_silence_threshold')
    hotwords: Optional[str] = w_config.get('hotwords')
    language_detection_threshold: Optional[float] = w_config.get('language_detection_threshold')
    language_detection_segments: int = w_config.get('language_detection_segments', 1)

class WhisperTranscriber:
    def __init__(self, model_name, device, compute_type):
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_file_path: str, options: dict) -> str:
        segments, _ = self.model.transcribe(audio_file_path, **options)
        transcription = " ".join([segment.text for segment in segments])
        return transcription

transcriber = WhisperTranscriber(
    model_name=w_config['model_name'],
    device=w_config['device'],
    compute_type=w_config['compute_type']
)

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), options: TranscriptionOptions = Depends()):
    temp_audio_file_path = None
    converted_file_path = None
    was_converted = False
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_audio_file:
            temp_audio_file.write(await file.read())
            temp_audio_file_path = temp_audio_file.name

        converted_file_path, was_converted = convert_to_wav(temp_audio_file_path)

        transcription = transcriber.transcribe(converted_file_path, options.dict(exclude_none=True))

        return {"transcription": transcription}
    finally:
        if temp_audio_file_path and os.path.exists(temp_audio_file_path):
            os.remove(temp_audio_file_path)
        if was_converted and converted_file_path and os.path.exists(converted_file_path):
            os.remove(converted_file_path)

def create_ui():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Whisper Transcription</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background-color: #f8f9fa;
            }
            .container {
                max-width: 700px;
            }
            #transcriptionOutput {
                white-space: pre-wrap;
                word-wrap: break-word;
            }
        </style>
    </head>
    <body>
        <div class="container mt-5">
            <div class="card">
                <div class="card-body">
                    <h1 class="card-title text-center mb-4">Upload Audio for Transcription</h1>
                    <div class="mb-3">
                        <input class="form-control" type="file" id="audioFile" accept="audio/*">
                    </div>
                    <div class="d-grid">
                        <button class="btn btn-primary" onclick="transcribeAudio()">
                            <span class="spinner-border spinner-border-sm d-none" role="status" aria-hidden="true" id="spinner"></span>
                            Transcribe
                        </button>
                    </div>
                    <h2 class="mt-4">Transcription:</h2>
                    <div class="p-3 bg-light rounded">
                        <pre id="transcriptionOutput"></pre>
                    </div>
                </div>
            </div>
        </div>

        <script>
            async function transcribeAudio() {
                const fileInput = document.getElementById('audioFile');
                const file = fileInput.files[0];
                if (!file) {
                    alert("Please select a file first.");
                    return;
                }

                const formData = new FormData();
                formData.append('file', file);

                const outputElement = document.getElementById('transcriptionOutput');
                const spinner = document.getElementById('spinner');
                const transcribeButton = document.querySelector('button');

                outputElement.innerText = '';
                spinner.classList.remove('d-none');
                transcribeButton.disabled = true;


                try {
                    const response = await fetch('/transcribe', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const result = await response.json();
                        if (result.transcription) {
                            outputElement.innerText = result.transcription;
                        } else if (result.error) {
                            outputElement.innerText = 'Error: ' + result.error;
                        }
                    } else {
                        const errorText = await response.text();
                        outputElement.innerText = 'Error: ' + response.statusText + ' - ' + errorText;
                    }
                } catch (error) {
                    outputElement.innerText = 'An error occurred: ' + error;
                } finally {
                    spinner.classList.add('d-none');
                    transcribeButton.disabled = false;
                }
            }
        </script>
    </body>
    </html>
    '''

if __name__ == "__main__":
    import uvicorn

    s_config = config['server']

    if s_config['ui'] or "--ui" in sys.argv:
        @app.get("/", response_class=HTMLResponse)
        async def read_root():
            return create_ui()

    uvicorn.run(
        app,
        host=s_config['host'],
        port=s_config['port']
    )
