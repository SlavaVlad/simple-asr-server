#!/bin/bash

# Docker entrypoint script for ASR service
# Generates API keys if needed and starts the server

set -e

# Function to generate a secure API key
generate_api_key() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
    else
        echo ""
    fi
}

# Set default values for environment variables
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-9854}
export DEFAULT_MODEL=${DEFAULT_MODEL:-"turbo"}
export MODEL_DOWNLOAD_ROOT=${MODEL_DOWNLOAD_ROOT:-"/app/models"}
export KEYS_FILE=${KEYS_FILE:-"/app/data/keys.txt"}
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
        echo ""
        echo "Creating empty keys file - you must add keys manually before starting the server."
        touch "${KEYS_FILE}"
    fi
fi

echo "Starting Simple ASR Server in Docker..."
echo "Host: ${HOST}"
echo "Port: ${PORT}"
echo "Default Model: ${DEFAULT_MODEL}"
echo "Model Download Root: ${MODEL_DOWNLOAD_ROOT}"
echo "Keys File: ${KEYS_FILE}"
echo "Log Level: ${LOG_LEVEL}"

# Start the server
exec python app.py
