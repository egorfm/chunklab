FROM python:3.12-slim

# Prevent Python from writing .pyc and enable buffer-free logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies for PyMuPDF and ML libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Ensure data directories exist
RUN mkdir -p /app/data/uploads /app/data/qdrant_storage

EXPOSE 8000 8501

# Default command launches backend, overrideable in docker-compose
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
