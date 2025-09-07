#!/bin/bash

# Simple ASR Server startup script for systemd
# This script loads environment variables from .env file and starts the server

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="${SCRIPT_DIR}"

# Set ROCm environment variables if ROCm is available
if [ -d "/opt/rocm" ]; then
    export ROCM_PATH=${ROCM_PATH:-"/opt/rocm"}
    export PATH="${ROCM_PATH}/bin:${PATH}"
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
    # Set HIP_VISIBLE_DEVICES to use all available GPUs
    export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0"}
    echo "ROCm detected, configured environment variables"
fi

# Function to generate a secure API key
generate_api_key() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
    else
        echo ""
    fi
}

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
export MODEL_DEVICE=${MODEL_DEVICE:-"cuda"}
export MODEL_DOWNLOAD_ROOT=${MODEL_DOWNLOAD_ROOT:-"${APP_DIR}/models"}
export KEYS_FILE=${KEYS_FILE:-"${APP_DIR}/data/keys.txt"}
export LOG_LEVEL=${LOG_LEVEL:-"info"}

# Create necessary directories
mkdir -p "${MODEL_DOWNLOAD_ROOT}"
mkdir -p "$(dirname "${KEYS_FILE}")"

# Check if keys file exists, create with generated key if not
if [ ! -f "${KEYS_FILE}" ]; then
    echo "Creating default keys file..."

    # Try to generate a secure key with openssl
    GENERATED_KEY=$(generate_api_key)

    if [ -n "${GENERATED_KEY}" ]; then
        echo "${GENERATED_KEY}" > "${KEYS_FILE}"
        echo "Generated secure API key using openssl: ${GENERATED_KEY}"
        echo "Created keys file at: ${KEYS_FILE}"
    else
        echo "WARNING: openssl not found! Cannot generate secure API key."
        echo "Please manually add API keys to ${KEYS_FILE}"
        echo "Each key should be 64 hex characters (32 bytes) on a separate line."
        echo "Example key format: 0000000000000000000000000000000000000000000000000000000000000000"
        echo ""
        echo "Creating empty keys file - you must add keys manually before starting the server."
        touch "${KEYS_FILE}"
    fi
fi

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
echo "Model Device: ${MODEL_DEVICE}"
echo "Model Download Root: ${MODEL_DOWNLOAD_ROOT}"
echo "Keys File: ${KEYS_FILE}"
echo "Log Level: ${LOG_LEVEL}"

# Start the server
exec python app.py
