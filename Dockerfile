# Hugging Face Spaces Docker image for the PatentVision AI FastAPI backend.
# Keep backend/Dockerfile intact for standalone backend/ hosts such as Render.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp/build
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces runs containers as non-root UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    REMBG_HOME=/home/user/.rembg

WORKDIR /home/user/app

# Pre-bake U-2-Net weights into the image so cold wakes need no runtime model download.
RUN python -c "import rembg; rembg.new_session('u2net')"

COPY --chown=user:user backend/app ./app
COPY --chown=user:user backend/alembic ./alembic
COPY --chown=user:user backend/alembic.ini ./alembic.ini

RUN mkdir -p /home/user/app/storage

# Single worker on purpose: job slots and the rate limiter are in-process.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
