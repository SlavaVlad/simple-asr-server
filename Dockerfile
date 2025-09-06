# Use official Python image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including openssl
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    openssl \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Update pip
RUN pip install --upgrade pip

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Copy startup script for key generation
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

# Create directory for models and data
RUN mkdir -p /app/models /app/data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=9854

# Expose port
EXPOSE 9854

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9854/health || exit 1

# Run the application
ENTRYPOINT ["./docker-entrypoint.sh"]
