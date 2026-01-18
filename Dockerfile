FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

COPY backend/requirements.txt backend/requirements-custom-ai.txt backend/requirements-multimodal.txt /app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-custom-ai.txt && \
    pip install --no-cache-dir -r requirements-multimodal.txt

RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

COPY backend /app/backend
COPY frontend /app/frontend

RUN mkdir -p /app/checkpoints /app/training_data /app/generated_outputs /app/uploaded_faces

EXPOSE 8000

ENV PYTHONPATH=/app/backend:$PYTHONPATH

CMD ["python", "backend/main.py"]
