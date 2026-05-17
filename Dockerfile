# NVIDIA CUDA base image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python command available
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install PyTorch with CUDA 12.1
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
COPY main.py .
# ✅ Copy system prompts explicitly
COPY system_prompts ./system_prompts

# Expose port
EXPOSE 8000

# Model path (bind mount)
ENV MODEL_PATH=/models/Qwen2.5-VL-3B-Instruct

# Start FastAPI automatically
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
