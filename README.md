# Whisper Transcription API

This project provides a RESTful API for audio transcription using a Whisper model. The API is built with FastAPI and runs in a Docker container.

## Prerequisites

Before you begin, ensure you have the following installed:

*   [Docker](https://docs.docker.com/get-docker/)
*   [Docker Compose](https://docs.docker.com/compose/install/)

## Project Structure

```
.
├── app.py              # Main application file with FastAPI endpoint
├── docker-compose.yml  # Docker Compose configuration
├── Dockerfile          # Dockerfile for building the application image
├── model/              # Directory for Whisper model files
└── requirements.txt    # Python dependencies
```

## Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/SlavaVlad/simple-asr-server
    cd simple-asr-server
    ```
3.  **Add API keys:**

    Create a `keys.txt` file in the root of the project and add your API keys, one per line.

## Building and Running the Project

You can build and run the project using Docker Compose.

1.  **Build the Docker image:**

    ```bash
    docker-compose build
    ```

2.  **Run the container:**

    ```bash
    docker-compose up
    ```

    The application will be available at `http://0.0.0.0:9854`.

## API Endpoint

### POST /transcribe

This endpoint accepts an audio file and returns the transcription.

*   **URL:** `/transcribe`
*   **Method:** `POST`
*   **Headers:**
    *   `X-API-Key`: Your API key.
*   **Form Data:**
    *   `file`: The audio file to be transcribed.

**Example using `curl`:**

```bash
curl -X POST "http://localhost:9854/transcribe" \
     -H "X-API-Key: YOUR_API_KEY" \
     -F "file=@/path/to/your/audio.wav"
```

**Successful Response (200 OK):**

```json
{
  "transcription": [
    {
      "start_time": 0.0,
      "end_time": 2.5,
      "transcription": "Hello world."
    }
  ],
  "text": "Hello world. ",
  "metrics": {
    "processing_time": 5.2,
    "rtf": 0.5,
    "word_rate": 2.0
  }
}
```

**Error Response (401 Unauthorized):**

If the API key is missing or invalid.

```json
{
  "detail": "Invalid API Key"
}
```
