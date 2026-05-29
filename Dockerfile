# llama.cpp server (CUDA) + FastAPI proxy
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /service

COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY main.py .
COPY system_prompts ./system_prompts
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV LLAMA_API_BASE=http://127.0.0.1:8080/v1 \
    LLAMA_API_KEY=not-needed \
    MODEL_NAME=local-model \
    MODEL_FILE=/models/Qwen2.5-VL-3B-Instruct-GGUF/Qwen2.5-VL-3B-Instruct-Q8_0.gguf \
    MMPROJ_FILE=/models/Qwen2.5-VL-3B-Instruct-GGUF/mmproj-model-f16.gguf \
    LLAMA_HOST=0.0.0.0 \
    LLAMA_PORT=8080 \
    CTX_SIZE=16384 \
    N_THREADS=4 \
    N_GPU_LAYERS=-1

EXPOSE 8080 8000

ENTRYPOINT ["/entrypoint.sh"]
