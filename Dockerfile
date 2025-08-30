FROM python:3.11-slim

# System deps (ffmpeg + ffprobe)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (keeps assets/ for fonts, etc.)
COPY . .

# Default port (Railway will set PORT)
ENV PORT=8000
EXPOSE 8000

# Start FastAPI; use Railway's $PORT when present
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]