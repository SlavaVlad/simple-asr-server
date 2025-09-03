#!/bin/bash

# Simple ASR Server startup script for systemd
# This script loads environment variables from .env file and starts the server

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}"

# Load environment variables from .env file if it exists
if [ -f "${APP_DIR}/.env" ]; then
    echo "Loading environment variables from ${APP_DIR}/.env"
    set -a  # automatically export all variables
    source "${APP_DIR}/.env"
    set +a
else
    echo "Warning: .env file not found at ${APP_DIR}/.env"
    echo "Using default environment variables"
fi

# Set default values if not provided in .env
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-9854}
export DEFAULT_MODEL=${DEFAULT_MODEL:-"turbo"}
export MODEL_DOWNLOAD_ROOT=${MODEL_DOWNLOAD_ROOT:-"${APP_DIR}/models"}
export KEYS_FILE=${KEYS_FILE:-"${APP_DIR}/keys.txt"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# Create necessary directories
mkdir -p "${MODEL_DOWNLOAD_ROOT}"
mkdir -p "$(dirname "${KEYS_FILE}")"

# Check if virtual environment exists, create if not
VENV_DIR="${APP_DIR}/venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Install/upgrade dependencies
echo "Installing/upgrading dependencies..."
pip install --upgrade pip
pip install -r "${APP_DIR}/requirements.txt"

# Change to app directory
cd "${APP_DIR}"

echo "Starting Simple ASR Server..."
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Default Model: ${DEFAULT_MODEL}"
echo "Model Download Root: ${MODEL_DOWNLOAD_ROOT}"
echo "Keys File: ${KEYS_FILE}"
echo "Log Level: ${LOG_LEVEL}"

# Start the application
exec python3 app.py

